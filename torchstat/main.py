import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import abc
from torch import nn
from torchstat import stat, ModelStat,StatNode,StatTree
from torchstat.reporter import round_value
import traceback
#If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

# The ID and range of a sample spreadsheet.
SAMPLE_SPREADSHEET_ID = '1gaDAHXNkthmyBFkFWlRmrX1YxPAaJq4tz9F-H_YzXI0'
SAMPLE_RANGE_NAME = 'VAE-VAE:[flops]'

def init_googleapi_credentials(json_dir_path:str):
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    token_json_path = os.path.join(json_dir_path, "token.json")
    if os.path.isfile(token_json_path):
      creds = Credentials.from_authorized_user_file(token_json_path, SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        print("[INFO] No valid credentials found. Please login.")
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            cred_path = os.path.join(json_dir_path, "credentials.json")
            flow = InstalledAppFlow.from_client_secrets_file(
               cred_path, SCOPES
            )
            creds = flow.run_local_server(port=0)
    # Save the credentials for the next run
        with open(token_json_path, "w") as token:
            token.write(creds.to_json())
    return creds
def create_empty_list(save_list:list[list], depth:int):
    for i in range(depth + 1):
        save_list.append([])
    return save_list
def traverse_stat_tree(cur_node:StatNode, save_list:list[list], depth:int, attr:str, max_depth:int):
    attr_value = round_value(getattr(cur_node, attr))
    #print(f"[INFO] Traversing {cur_node._name} , leaf_children: {cur_node.num_leaf_children}, attr value: {attr_value}, depth: {depth}")
    if cur_node.is_leaf or depth >= max_depth:
        # should return!
        # but record first    
        save_list[depth].append(f"{cur_node._name}:{attr_value}")
        for i in range(depth+1, max_depth+1):
            save_list[i].append("")
        return 1
    sum_len = 0
    for child in cur_node.children:
        sum_len += traverse_stat_tree(child, save_list, depth + 1, attr, max_depth)
    #print(f"[INFO] Sum length of {cur_node._name} is {sum_len}")
    node_name_position = (sum_len - 1) // 2 #  0-based list
    cur_len = len(save_list[depth])
    save_list[depth] = save_list[depth] + [""] * (sum_len)
    save_list[depth][cur_len + node_name_position] = f"{cur_node._name}:{attr_value}"
    return sum_len
def convert_column_to_int(column:str) -> int:
    """
    for example, 'A' -> 1, 'AA' -> 28, 'AB' -> 29, 'BA' -> 55
    """
    sum = 0
    for i in reversed(range(len(column))):
        sum += sum * 27 + ord(column[i]) - ord('A') + 1
    return sum
def convert_int_to_column(num:int) -> str:
    """
    for example, 1 -> 'A', 28 -> 'AA', 29 -> 'AB', 55 -> 'BA'
    """
    column = ""
    while num > 0:
        column = chr(num % 27 - 1 + ord('A')) + column
        num = num // 27
    return column
def get_valid_column(st_column:str, length:int) -> str:
    """
    for example, 'A' + 27 = 'AA', 'A' + 27 + 1 = 'AB', 'AB' + 27 = 'BB'
    """
    st_colum_int = convert_column_to_int(st_column)
    end_column_int = st_colum_int + length
    
    end_colum_str = convert_int_to_column(end_column_int)
    print(f"[INFO] Get valid column from {st_column}: {st_colum_int} + {length} = {end_column_int} -> {end_colum_str}")
    return end_colum_str
    
class StatUpdate(abc.ABC):
    def __init__(self, spreadsheet_id:str, sheet_id:str ,start_range:str, creds: Credentials = None, value_input_option:str = "USER_ENTERED"):
        if not creds or not creds.valid:
            print("[INFO] No valid credentials provided. Try load from token.json.")
            creds = init_googleapi_credentials()
        self.creds = creds
        self.service = build("sheets", "v4", credentials=creds)
        self.spreadsheet_id = spreadsheet_id
        self.range_column = start_range[0]
        self.value_input_option = value_input_option
        assert self.range_column.isalpha()
        self.sheet_id = sheet_id
        self.range_row = start_range[1]
        assert self.range_row.isdigit()
        self.attr_dict = {
            "_parameter_quantity": None,
            "_Flops": None
        }
        print(f"[INFO] Google API service created with spreadsheet_id: {spreadsheet_id}, start_range: {start_range}")
    def update(self, model:nn.Module, input_shape:tuple ,query_granularity : int = 1):
        try:
            model_stat = ModelStat(model, input_shape, query_granularity)
            model_stat._analyze_model()
            self.stat_tree = model_stat.stat_tree
            self.str_stat, self.df_stat = model_stat.show_report()
        except Exception as e:
            traceback.print_exc()
            print(f"[ERROR] {e}")
            return
        print("[INFO] Stats computed.")
    def get_model_and_input_shape(self):
        return nn.Identity(), (3, 256, 256)
    def parse_info(self, max_depth = 5):
        """
            This function parse precomputed stats in self.df_stats, and return a values that can be used to update the google sheet.
            in df_stat, we currently care about three columns:
            1. module name
            2. params
            3. flops
            
            For a module with name X, X is named by the rule:

            X : Name.index.X
            
            here Name is the name of the highest level module(list) in the class, index is the index of the module in the list, and X is the rest of the module name.
            
            We call "Name.index" a layer. For example, "down.3.mid.4.conv1" has a height of 3 levels(the last conv1 is seemed as a level).

            The returned value should a list (of list), designed as follows:
                         params                       flops
                            down:sum                  The same
            1           |        2:sum |    3| 
            mid
            1 |2 | 3| 4 |
            conv1
            
        """
        assert hasattr(self, "stat_tree")
        root_node = self.stat_tree.root_node
        root_max_height = root_node._depth
        for attr in self.attr_dict.keys():
            print(f"[INFO] Parsing {attr}")
            save_list = create_empty_list([], max_depth)
            traverse_stat_tree(root_node, save_list, 0, attr, max_depth)
            self.attr_dict[attr] = save_list
    def __repr__(self) -> str:
        if not hasattr(self, "str_stats"):
            return "No stats computed."
        return self.str_stats
    def update_to_googlesheet(self):
        for key in self.attr_dict.keys():
            _values = self.attr_dict[key]
            if _values is None:
                continue
            body = {"values": _values}
            length = len(_values)
            max_depth = len(_values[0])
            end_column = get_valid_column(self.range_column, max_depth + 1)
            end_row = int(self.range_row) + length + 1
            range_name = f"{self.sheet_id}!{self.range_column}{self.range_row}:{end_column}{end_row}"
            self.range_row = end_row + 1
            print(f"[INFO] Updating {key} to {self.sheet_id}!{range_name}")
            try:
                result = (
                    self.service.spreadsheets()
                    .values()
                    .update(
                        spreadsheetId=self.spreadsheet_id,
                        range=range_name,
                        valueInputOption=self.value_input_option,
                        body=body,
                    )
                    .execute()
                )
            except HttpError as error:
                print(f"An error occurred when updating {key}: {error}")
                return error
        print('[INFO] Update finished.')
    def run(self):
        self.update(self.get_model_and_input_shape())
        self.parse_info()
        self.update_to_googlesheet()
class nested_model(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(64, 64, 3),
            nn.GroupNorm(32, 64),
        )
        self.conv = nn.Conv2d(64, 128, 3)
        self.bn = nn.BatchNorm2d(128)
    def forward(self, x):
        x = self.seq(x)
        x = self.conv(x)
        x = self.bn(x)
        return x
class test_model(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.GroupNorm(32, 64),
        )
        self.nested = nested_model()
        self.bn = nn.BatchNorm2d(128)
    def forward(self, x):
        x = self.seq(x)
        x = self.nested(x)
        x = self.bn(x)
        return x           
import sys
sys.setrecursionlimit(1500) # compromise for bad implementation of torchstat

def nested_test():
    creds = init_googleapi_credentials()
    stat_update = StatUpdate(SAMPLE_SPREADSHEET_ID, 'B2', creds)
    test_example = test_model()
    stat_update.update(test_example, (3, 256, 256))
    stat_update.parse_info(4)
    print(stat_update.attr_dict)
    stat_update.update_to_googlesheet()
def main():
    spreadsheet_id = sys.argv[1]
    sheet_id = sys.argv[2]
    json_dir_path = sys.argv[3]
    assert os.path.isdir(json_dir_path)
    start_range = sys.argv[4] if len(sys.argv) > 4 else 'B2'
    creds = init_googleapi_credentials(json_dir_path)
    stat_update = StatUpdate(spreadsheet_id, sheet_id , start_range, creds)
    test_example = VQVAE()
    #test_example = test_model()
    stat_update.update(test_example, (3, 256, 256))
    stat_update.parse_info(4)
    print(stat_update.attr_dict)
    stat_update.update_to_googlesheet()

if __name__ == "__main__":
    main()