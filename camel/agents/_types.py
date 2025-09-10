# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
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
# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
from typing import Any, Dict, List, Optional, Union

from openai import AsyncStream, Stream
from openai.types.chat import ChatCompletionChunk
from pydantic import BaseModel, ConfigDict

from camel.messages import BaseMessage
from camel.types import ChatCompletion


class ToolCallRequest(BaseModel):
    r"""The request for tool calling."""

    tool_name: str  # 工具名称
    args: Dict[str, Any]  # 参数
    tool_call_id: str  # 工具调用ID


class ModelResponse(BaseModel):
    r"""The response from the model."""

    model_config = ConfigDict(arbitrary_types_allowed=True)  # 允许任意类型的参数
    response: Union[
        ChatCompletion,
        Stream[ChatCompletionChunk],
        AsyncStream[ChatCompletionChunk],
    ]  # 响应类型
    tool_call_requests: Optional[List[ToolCallRequest]]  # 工具调用请求
    output_messages: List[BaseMessage]  # 输出消息
    finish_reasons: List[str]  # 完成原因
    usage_dict: Dict[str, Any]  # 使用字典
    response_id: str  # 响应ID
