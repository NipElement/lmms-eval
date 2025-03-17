import base64
import json
import os
import time
from io import BytesIO
from typing import List, Optional, Tuple, Union

import decord
# 修改导入方式，避免命名冲突
import requests as http_requests
from accelerate import Accelerator
from loguru import logger as eval_logger
from PIL import Image
import base64
from io import BytesIO
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import read_video_pyav_base64

def encode_base64_image(image:Image.Image, image_format="PNG") -> str:
    im_file = BytesIO()
    image.save(im_file, format=image_format)
    im_bytes = im_file.getvalue()
    im_64 = base64.b64encode(im_bytes).decode("utf-8")
    return im_64

def encode_base64_image_url(image:Image.Image, image_format="PNG") -> str:
    return f"data:image/{image_format};base64,{encode_base64_image(image, image_format)}"

@register_model("qwen2_5_vl_interleave_api")
class Qwen2_5_VL_Interleave_API(lmms):
    """
    Qwen2.5_VL Model using vLLM API with interleaved text and images
    """

    def __init__(
        self,
        api_url: str = "http://127.0.0.1:8000/v1/chat/completions",
        api_key: Optional[str] = None,
        model_name: str = "/map-vepfs/huggingface/models/Qwen2.5-VL-7B-Instruct",
        batch_size: Optional[Union[int, str]] = 1,
        max_num_frames: int = 32,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[float] = None,
        max_image_size: Optional[int] = None,
        image_token: str = "<image>",  # 用于分隔交错的图像和文本
        system_prompt: str = "You are a helpful assistant.",
        continual_mode: bool = False,
        response_persistent_folder: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self.api_url = api_url
        self.api_key = api_key if api_key else os.environ.get("OPENAI_API_KEY", None)
        self.model_name = model_name
        self.use_custom_video_loader = use_custom_video_loader
        self.fps = fps
        self.max_image_size = max_image_size
        self.max_num_frames = max_num_frames
        self.image_token = image_token
        self.system_prompt = system_prompt

        accelerator = Accelerator()
        self._rank = accelerator.local_process_index
        self._world_size = accelerator.num_processes
        self._device = accelerator.device
        self.batch_size_per_gpu = int(batch_size)

        # 设置持续模式（如需要）
        self.continual_mode = continual_mode
        if self.continual_mode and response_persistent_folder is None:
            raise ValueError("Continual mode requires a persistent path for the response. Please provide a valid path.")
        
        if self.continual_mode:
            os.makedirs(response_persistent_folder, exist_ok=True)
            self.response_persistent_folder = response_persistent_folder
            self.model_version = self.model_name.split("/")[-1].replace("-", "_").lower()
            self.response_persistent_file = os.path.join(self.response_persistent_folder, f"{self.model_version}_api_response.json")

            if os.path.exists(self.response_persistent_file):
                with open(self.response_persistent_file, "r") as f:
                    self.response_cache = json.load(f)
                self.cache_mode = "resume"
            else:
                self.response_cache = {}
                self.cache_mode = "start"

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size
    
    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Qwen2.5_VL API")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def get_uuid(task, split, doc_id):
            return f"{task}___{split}___{doc_id}"

        def _collate(x):
            # 按文本长度降序排序
            return -len(x[0]), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        # 按生成参数分组请求
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)

            task = task[0]
            split = split[0]
            
            # 检查缓存（如果启用了continual_mode）
            if self.continual_mode and self.cache_mode == "resume":
                cached_content = []
                all_cached = True
                for idx, id_val in enumerate(doc_id):
                    doc_uuid = get_uuid(task, split, id_val)
                    if doc_uuid in self.response_cache:
                        content = self.response_cache[doc_uuid]
                        if content:
                            cached_content.append(content)
                        else:
                            all_cached = False
                            break
                    else:
                        all_cached = False
                        break
                
                if all_cached:
                    res.extend(cached_content)
                    pbar.update(len(cached_content))
                    continue

            # 处理视觉内容
            visuals = [doc_to_visual[i](self.task_dict[task][split][ids]) for i, ids in enumerate(doc_id)]
            gen_kwargs = all_gen_kwargs[0]

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            # 规范化图像标记
            for i in range(len(contexts)):
                for j in range(32):
                    if f"<image_{j}>" in contexts[i]:
                        contexts[i] = contexts[i].replace(f"<image_{j}>", self.image_token)

            for i, context in enumerate(contexts):
                # 获取视觉内容
                visual = visuals[i] if i < len(visuals) else None
                
                # 准备消息格式
                message = [{"role": "system", "content": self.system_prompt}]
                
                if visual:
                    # 准备图像内容
                    if isinstance(visual, Image.Image):
                        visual = [visual]
                    
                    # 处理视频文件
                    if isinstance(visual, (list, tuple)) and isinstance(visual[0], str) and visual[0].endswith((".mp4", ".avi", ".mov")):
                        visual = visual[0]  # 获取视频路径
                        # 处理视频
                        if self.use_custom_video_loader:
                            try:
                                frames = read_video_pyav_base64(
                                    visual,
                                    num_frm=self.max_num_frames,
                                    fps=self.fps,
                                    img_format="JPEG",
                                    max_image_size=self.max_image_size,
                                )
                                images = list(map(lambda x: f"data:image/jpeg;base64,{x}", frames))
                            except Exception as e:
                                eval_logger.error(f"Failed to load video: {visual}. Error: {e}")
                                images = []
                        else:
                            vr = decord.VideoReader(visual)
                            images = []
                            for idx in range(min(len(vr), self.max_num_frames)):
                                frame = vr[idx].asnumpy()
                                img = Image.fromarray(frame)
                                buffer = BytesIO()
                                img.save(buffer, format="JPEG")
                                base64_bytes = base64.b64encode(buffer.getvalue())
                                base64_string = base64_bytes.decode("utf-8")
                                images.append(f"data:image/jpeg;base64,{base64_string}")

                        # 使用交错方式处理视频帧
                        user_content = []
                        user_content.append({"type": "video", "video": images})
                        user_content.append({"type": "text", "text": context})
                        message.append({"role": "user", "content": user_content})
                        
                    # 处理单张或多张图片
                    elif isinstance(visual, str) and visual.endswith((".jpg", ".jpeg", ".png")) or isinstance(visual, Image.Image) or (isinstance(visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in visual)):
                        # 处理单张图片
                        if isinstance(visual, str):
                            img = Image.open(visual).convert("RGB")
                            buffer = BytesIO()
                            img.save(buffer, format="JPEG")
                            base64_bytes = base64.b64encode(buffer.getvalue())
                            base64_string = base64_bytes.decode("utf-8")
                            images = [f"data:image/jpeg;base64,{base64_string}"]
                            
                        elif isinstance(visual, Image.Image):
                            img = visual.convert("RGB")
                            buffer = BytesIO()
                            img.save(buffer, format="JPEG")
                            base64_bytes = base64.b64encode(buffer.getvalue())
                            base64_string = base64_bytes.decode("utf-8")
                            images = [f"data:image/jpeg;base64,{base64_string}"]
                            
                        else:  # 多张图片的情况
                            images = []
                            for img in visual:
                                img = img.convert("RGB")
                                buffer = BytesIO()
                                img.save(buffer, format="JPEG")
                                base64_bytes = base64.b64encode(buffer.getvalue())
                                base64_string = base64_bytes.decode("utf-8")
                                images.append(f"data:image/jpeg;base64,{base64_string}")
                        
                        # 使用交错模式处理图像和文本
                        if self.image_token not in context:
                            # 如果没有图像标记，将图像放在文本前面
                            user_content = []
                            for img_url in images:
                                user_content.append({"type": "image_url", "image_url": {"url": image_url}})
                            user_content.append({"type": "text", "text": context})
                            message.append({"role": "user", "content": user_content})
                        else:
                            # 如果有图像标记，按标记分割文本并交错插入图像
                            text_parts = context.split(self.image_token)
                            
                            # 确保图像数量与分割数量匹配
                            if len(images) != len(text_parts) - 1:
                                eval_logger.warning(f"Number of images ({len(images)}) and context parts ({len(text_parts)}) don't match. Proceeding with available images.")
                            
                            # 构建交错内容
                            user_content = []
                            for j in range(len(text_parts)):
                                if text_parts[j]:
                                    user_content.append({"type": "text", "text": text_parts[j]})
                                if j < len(images) and j < len(text_parts) - 1:
                                    user_content.append({"type": "image_url", "image_url": {"url": images[j]}})
                            
                            message.append({"role": "user", "content": user_content})
                    else:
                        # 没有视觉内容或不支持的类型
                        message.append({"role": "user", "content": [{"type": "text", "text": context}]})
                else:
                    # 纯文本情况
                    message.append({"role": "user", "content": [{"type": "text", "text": context}]})                
                # 设置API请求参数
                if "max_new_tokens" not in gen_kwargs:
                    gen_kwargs["max_new_tokens"] = 1024
                if "temperature" not in gen_kwargs:
                    gen_kwargs["temperature"] = 0
                if "top_p" not in gen_kwargs:
                    gen_kwargs["top_p"] = 1.0

                # 准备API调用
                request_body = {
                    "model": self.model_name,
                    "messages": message,
                    "max_new_tokens": gen_kwargs["max_new_tokens"],
                    "temperature": gen_kwargs["temperature"],
                    "top_p": gen_kwargs["top_p"],
                    "stream": False
                }
                response = http_requests.post(
                    self.api_url,
                    json=request_body
                )
                response.raise_for_status()
                response_data = response.json()
                answer = response_data['choices'][0]['message']['content']
                
                # 保存结果
                res.append(answer)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), answer)
                
                # 如果启用了持续模式，缓存响应
                if self.continual_mode:
                    doc_uuid = get_uuid(task, split, doc_id[i])
                    self.response_cache[doc_uuid] = answer
                    with open(self.response_persistent_file, "w") as f:
                        json.dump(self.response_cache, f)
                        
                pbar.update(1)

        # 恢复原始顺序
        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation is not implemented for Qwen2.5_VL API")
