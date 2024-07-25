from transformers import CLIPProcessor, CLIPModel, CLIPImageProcessor, CLIPTokenizer
from PIL import Image
import torch
from torchvision import transforms
def get_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor
def differentiable_process(processor: CLIPProcessor, image:Image.Image) -> torch.Tensor:
    image_mean, image_std = processor.image_processor.image_mean, processor.image_processor.image_std
    image_size = processor.image_processor.size['shortest_edge']
    image = image.convert("RGB").resize((image_size, image_size))
    mean_tensor, std_tensor = torch.tensor(image_mean).view(1, 3, 1, 1), torch.tensor(image_std).view(1, 3, 1, 1)
    to_tensor = transforms.PILToTensor()
    image = to_tensor(image).unsqueeze(0)
    return (image - mean_tensor) / std_tensor, mean_tensor, std_tensor
def one_forward(model: CLIPModel, processor: CLIPProcessor, text, image: torch.Tensor) -> torch.Tensor:
    inputs:torch.Tensor = processor(text, return_tensors='pt', padding=True).input_ids
    output = model(input_ids=inputs, pixel_values=image)
    logits_per_image = output.logits_per_image
    return logits_per_image

def PGD_attack(model, processor, text, image, epsilon=4./255, alpha=1., iters=40):
    image = image.convert("RGB").resize((224, 224))
    image_input = transforms.ToTensor()(image).unsqueeze(0).to(torch.float32) 
    print(image_input.max(), image_input.min())
    image_mean, image_std = processor.image_processor.image_mean, processor.image_processor.image_std
    mean_tensor, std_tensor = torch.tensor(image_mean).view(1, 3, 1, 1), torch.tensor(image_std).view(1, 3, 1, 1)
    original_image_input = image_input.detach().clone()
    original_image_input.requires_grad = False
    loss = []
    for i in range(iters):
        actual_lr = alpha * (i + 1) / iters
        image_input.requires_grad = True
        normalized_image_input = (image_input - mean_tensor) / std_tensor
        logits = one_forward(model, processor, text, normalized_image_input)
        prob = torch.softmax(logits, dim=1)
        print(prob)
        # calculate cross entropy loss = -log(prob[:,0])
        loss = -torch.log(prob[:,0]).mean()
        print(f"iter {i}, loss: {loss.item()}")
        loss.backward()
        grad = image_input.grad.sign().detach().clone()
        image_input = image_input + alpha * grad
        image_input = image_input.clamp(original_image_input - epsilon, original_image_input + epsilon)
        image_input = image_input.clamp(0, 1)
        image_input = image_input.detach()
    return image_input
def naive_attack(model, processor, text, image, epsilon=4./255., alpha=1., iters=40, noise_sigma=0.):
    image = image.convert("RGB").resize((224, 224))
    image_input = transforms.ToTensor()(image).unsqueeze(0).to(torch.float32) 
    print(image_input.max(), image_input.min())
    image_mean, image_std = processor.image_processor.image_mean, processor.image_processor.image_std
    mean_tensor, std_tensor = torch.tensor(image_mean).view(1, 3, 1, 1), torch.tensor(image_std).view(1, 3, 1, 1)
    original_image_input = image_input.detach().clone()
    original_image_input.requires_grad = False
    losses = []
    for i in range(iters):
        image_input.requires_grad = True
        normalized_image_input = (image_input - mean_tensor) / std_tensor
        logits = one_forward(model, processor, text, normalized_image_input)
        prob = torch.softmax(logits, dim=1)
        print(prob)
        # calculate cross entropy loss = -log(prob[:,0])
        loss = -torch.log(prob[:,0]).mean()
        losses.append(loss.item())
        loss.backward()
        grad = image_input.grad.detach().clone()
        if noise_sigma > 0:
            grad = grad + torch.randn_like(grad) * grad.norm(p=2) * noise_sigma
        image_input = image_input + alpha * grad
        image_input = image_input.clamp(original_image_input - epsilon, original_image_input + epsilon)
        image_input = image_input.clamp(0, 1)
        image_input = image_input.detach()
    return image_input, losses
from matplotlib import pyplot as plt
import numpy as np
def main():
    model, processor = get_model()
    text = ["a photo of a cat", "a photo of a dog"]
    image = Image.open("./visuals/cat.jpg")
    #adv_image = PGD_attack(model, processor, text, image)
    adv_image, losses = naive_attack(model, processor, text, image, noise_sigma=0.)
    adv_image_2, losses_1 = naive_attack(model, processor, text, image, noise_sigma=0.01)
    adv_image_2, losses_2 = naive_attack(model, processor, text, image, noise_sigma=0.1)
    adv_image_3, losses_3 = naive_attack(model, processor, text, image, noise_sigma=1)
    # log scale instead of linear scale
    losses = np.log(np.array(losses) + 1e-6)
    losses_1 = np.log(np.array(losses_1) + 1e-6)
    losses_2 = np.log(np.array(losses_2) + 1e-6)
    losses_3 = np.log(np.array(losses_3) + 1e-6)
    plt.plot(losses, label="noise_sigma=0")
    plt.plot(losses_1, label="noise_sigma=0.01")
    plt.plot(losses_2, label="noise_sigma=0.1")
    plt.plot(losses_3, label="noise_sigma=1")
    plt.xlabel("gradient ascent iterations")
    plt.ylabel("log(CE loss)")
    plt.legend()
    plt.savefig("./visuals/losses.jpg")
    adv_image = adv_image.squeeze(0).detach().cpu().numpy()
    adv_image = adv_image.transpose(1, 2, 0)
    adv_image = (adv_image * 255).astype("uint8")
    adv_image = Image.fromarray(adv_image)
    adv_image.save("./visuals/adv_cat.jpg")

if __name__ == "__main__":
    main()