## Tool: Extract Model Statistics to Google Sheet

**This repo allows you to extract information such as Flops, parameters from an arbitrary nn.Module and automatically push to google sheet**

Usage:

```python
from main import StatUpdate
# inherit StatUpdate class and reload the abstract method get_model_and_input_shape
# this method should return the desired model you want to test and input shape

Instance = YourClass(spreadsheet_id, sheet_name, credential)
Instance.run()
```