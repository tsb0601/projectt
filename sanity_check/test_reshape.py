from git import Union
import torch
def print_square(x:torch.Tensor):
    # x is of shape [H,W]
    # print it in form
    """
    ------
    |x_00|x_01|...
    ----
    |x_10|x_11|...
    ...
    -----
    """
    H,W = x.shape
    print('')
    for i in range(H):
        print('-'*10)
        for j in range(W):
            print(f'|{x[i,j]:2d}', end='')
        print('')
    print('-'*10,'\n')
def L_to_P(zs:torch.Tensor, split:float = 1)-> torch.Tensor:
    """
    zs: [batch_size, seq_len, hidden_size]
    return: [batch_size, hidden_size//split, sqrt(seq_len*split), sqrt(seq_len*split)]
    """
    # reshape it to square
    batch_size, num_patches, hidden_size = zs.shape
    pn = int(num_patches ** 0.5)
    zs = zs.view(batch_size, pn, pn, hidden_size)
    #zs = self.forward_norm(zs)
    # channel goes first
    zs = zs.permute(0,3,1,2).contiguous() # [batch_size, hidden_size, patch_size, patch_size]
    sqrt_split = int(split ** 0.5)
    split_c = int(hidden_size // split)
    split_pn = pn * sqrt_split
    # reshape to bsz, split_c, split_pn, split_pn
    # first split to split_c, sqrt_split, sqrt_split, pn, pn
    zs = zs.view(batch_size, split_c, sqrt_split, sqrt_split, pn, pn)
    # then permute to split_c, split_pn, sqrt_split, split_pn, sqrt_split
    zs = zs.permute(0,1,4,2,5,3).contiguous()
    # then reshape to bsz, hidden_size, split_pn, split_pn
    zs = zs.reshape(batch_size, split_c, split_pn, split_pn)
    return zs.contiguous()
def P_to_L(zs:torch.Tensor, split:float = 1) -> torch.Tensor:
    """
    zs: [batch_size, hidden_size//split, sqrt(seq_len*split), sqrt(seq_len*split)]
    return: [batch_size, seq_len, hidden_size]
    """
    batch_size, c , pn, pn = zs.shape
    aggregated_c = c * split
    sqrt_split = int(split ** 0.5)
    split_pn = int(pn // sqrt_split)
    #zs = zs.view(batch_size, c, sqrt_split, split_pn, sqrt_split, split_pn)
    zs = zs.reshape(batch_size, c, split_pn, sqrt_split, split_pn, sqrt_split)
    #try reshape back to see diff
    # do a reverse permute to (0,1,4,2,5,3)
    zs = zs.permute(0,1,3,5,2,4).contiguous()
    zs = zs.view(batch_size, aggregated_c, split_pn, split_pn)
    zs = zs.permute(0,2,3,1).contiguous()
    zs = zs.view(batch_size, split_pn, split_pn, aggregated_c)
    zs = zs.view(batch_size, split_pn*split_pn, aggregated_c)
    return zs.contiguous()
def P_to_P(zs:torch.Tensor, split:float = 1)-> torch.Tensor:
    """
    zs: [batch_size, hidden_size, patch_size, patch_size]
    return: [batch_size, hidden_size//split, sqrt(patch_size*split), sqrt(patch_size*split)]
    """
    batch_size, hidden_size, pn, _ = zs.shape
    sqrt_split = int(split ** 0.5)
    split_c = int( hidden_size//split )
    split_pn = pn * sqrt_split
    original_zs = zs.clone()
    # reshape to bsz, split_c, split_pn, split_pn
    # first split to split_c, sqrt_split, sqrt_split, pn, pn
    zs = zs.view(batch_size, split_c, sqrt_split, sqrt_split, pn, pn)
    #try reshape back to see diff
    reshape_back_zs = zs.view(batch_size, hidden_size, pn, pn)
    # then permute to split_c, split_pn, sqrt_split, split_pn, sqrt_split
    zs = zs.permute(0,1,4,2,5,3).contiguous()
    # then reshape to bsz, hidden_size, split_pn, split_pn
    zs = zs.view(batch_size, split_c, split_pn, split_pn)
    return zs.contiguous()

def test_P_L_P(x:torch.Tensor, split:float = 1)->Union[bool, float]:
    """
    x: [batch_size, seq_len, hidden_size]
    """
    y = L_to_P(x, split)
    #print_square(y[0,0])
    z = P_to_L(y, split)
    #print_square(z[0])
    if torch.allclose(x,z,atol=1e-6):
        return True
    else:
        return torch.mean(torch.abs(x-z).float()).item()
def test_L_P_L(x:torch.Tensor, split:float = 1)->Union[bool, float]:
    """
    x: [batch_size, hidden_size//split, sqrt(seq_len*split), sqrt(seq_len*split)]
    """
    y = P_to_L(x, split)
    z = L_to_P(y, split)
    if torch.allclose(x,z,atol=1e-6):
        return True
    else:
        return torch.mean(torch.abs(x-z)).item()

def main():
    #P_L_P_x = torch.randn(1, 16, 4) # [batch_size, seq_len, hidden_size]
    P_L_P_x = torch.arange(1, 65).view(1, 16, 4)
    print_square(P_L_P_x[0])
    P_L_P_split = 1
    print(test_P_L_P(P_L_P_x, P_L_P_split))
    L_P_L_x = L_to_P(P_L_P_x, 1)
    #L_P_L_x = torch.randn(4, 64, 28, 28) # [batch_size, hidden_size//split, sqrt(seq_len*split), sqrt(seq_len*split)]
    L_P_L_split = 1
    print(test_L_P_L(L_P_L_x, L_P_L_split))
if __name__ == '__main__':
    main()