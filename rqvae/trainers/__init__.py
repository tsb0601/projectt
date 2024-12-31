# Copyright (c) 2022-present, Kakao Brain Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .trainer_stage1 import Trainer as TrainerStage1
from .trainer_stage2 import Trainer as TrainerStage2

STAGE1_ARCH_TYPE = [
    'rq-vae','dummy'
]


def create_trainer(config):
    if config.arch.stage == 1:
        return TrainerStage1
    else: 
        return TrainerStage2
