from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
from .base import LLMBase
import os, json


class CasualLM(LLMBase):
    """Huggingface Casual Language Models.

    Parameters:
    - model_path (str): The path/name for the desired language model.
    - arch (str, optional): The model architecture if different from model_path.
    - lora_path (str, optional): LoRA adapter path to load on top of base model.
    - merge_lora (bool): Whether to merge LoRA weights into the base model.
    - use_vllm (bool): Whether to use vLLM for inference.
    - max_tokens (int): Maximum number of tokens to generate.
    """

    def __init__(
        self,
        model_path=None,
        arch=None,
        use_vllm=False,
        max_tokens=2048,
        lora_path=None,
        merge_lora=True,
    ):
        self.arch = arch if arch is not None else model_path
        self.lora_path = lora_path if lora_path else None
        self.merge_lora = merge_lora
        self.tokenizer_use_fast = True
        self.max_tokens = max_tokens
        self.use_vllm=use_vllm
        self.lora_request = None
        super().__init__(model_path=model_path)

    def _infer_lora_rank(self):
        if self.lora_path is None:
            return None

        adapter_config_path = os.path.join(self.lora_path, "adapter_config.json")
        if not os.path.isfile(adapter_config_path):
            return None

        try:
            with open(adapter_config_path, "r", encoding="utf-8") as f:
                adapter_config = json.load(f)
            rank = adapter_config.get("r")
            rank_pattern = adapter_config.get("rank_pattern", {})
            if isinstance(rank_pattern, dict) and rank_pattern:
                pattern_ranks = [int(v) for v in rank_pattern.values()]
                rank = max(int(rank) if rank is not None else 0, max(pattern_ranks))
            if rank is not None:
                return int(rank)
        except Exception as e:
            print(f"> Failed to infer LoRA rank from '{adapter_config_path}': {e}")
        return None

    def load_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        if self.use_vllm:
            from vllm import LLM

            tokenizer_source = self.arch
            llm_kwargs = {
                "model": model_path,
                "tokenizer": model_path,
                "gpu_memory_utilization": 0.9,
            }
            self.lora_request = None
            if self.lora_path is not None:
                from vllm.lora.request import LoRARequest

                llm_kwargs["enable_lora"] = True
                lora_rank = self._infer_lora_rank()
                llm_kwargs["max_lora_rank"] = max(8, lora_rank) if lora_rank is not None else 64
                self.lora_request = LoRARequest("adapter", 1, self.lora_path)
                try:
                    tokenizer = AutoTokenizer.from_pretrained(self.lora_path)
                    tokenizer_source = self.lora_path
                except Exception:
                    print(
                        f"> No tokenizer found in adapter '{self.lora_path}'. "
                        f"Falling back to base tokenizer '{self.arch}'."
                    )
                    tokenizer = AutoTokenizer.from_pretrained(self.arch)
            else:
                tokenizer = AutoTokenizer.from_pretrained(self.arch)

            self.model = LLM(
                **llm_kwargs,
            )

            self.tokenizer = tokenizer
        else:

            torch_dtype = torch.bfloat16
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                load_in_8bit=False,
                low_cpu_mem_usage=True,
                device_map="auto",
            ).eval()

            tokenizer_source = self.arch
            if self.lora_path is not None:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(self.lora_path)
                    tokenizer_source = self.lora_path
                except Exception:
                    print(
                        f"> No tokenizer found in adapter '{self.lora_path}'. "
                        f"Falling back to base tokenizer '{self.arch}'."
                    )
                    tokenizer = AutoTokenizer.from_pretrained(self.arch)
            else:
                tokenizer = AutoTokenizer.from_pretrained(self.arch)

            if self.lora_path is not None:
                from peft import PeftModel

                model.resize_token_embeddings(len(tokenizer))
                model = PeftModel.from_pretrained(model, self.lora_path)
                if self.merge_lora:
                    model = model.merge_and_unload()
                model = model.eval()

            tokenizer.padding_side = "left"
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            if (
                getattr(model, "generation_config", None) is not None
                and tokenizer.eos_token_id is not None
            ):
                model.generation_config.pad_token_id = tokenizer.eos_token_id

            self.model = model
            self.tokenizer = tokenizer
            self.lora_request = None

        if self.lora_path is None:
            print(
                f"> Loading the provided {self.arch} checkpoint from '{model_path}'."
            )
        else:
            if self.use_vllm:
                print(
                    f"> Loading base model '{model_path}' with vLLM LoRA adapter "
                    f"'{self.lora_path}' (tokenizer='{tokenizer_source}')."
                )
            else:
                print(
                    f"> Loading base model '{model_path}' with LoRA adapter '{self.lora_path}' "
                    f"(tokenizer='{tokenizer_source}')."
                )

    def query(self, prompt, **kwargs):
        return self.query_generation(prompt, **kwargs)
    
    @torch.no_grad()
    def query_generation(self, prompt, do_sample=False, temperature=0.0):
        try:
            if self.use_vllm:
                from vllm import SamplingParams

                sampling_params = SamplingParams(
                    max_tokens=self.max_tokens,
                    temperature=temperature if do_sample else 0.0,
                )
                generate_kwargs = {}
                if self.lora_request is not None:
                    generate_kwargs["lora_request"] = self.lora_request
                outputs = self.model.generate(
                    [prompt], sampling_params,
                    **generate_kwargs,
                )
                pred = outputs[0].outputs[0].text
            else:
                gen_kwargs = {"max_new_tokens": self.max_tokens, "do_sample": do_sample}
                if do_sample:
                    gen_kwargs["temperature"] = temperature

                if self.model_path in [
                    "deepseek-ai/deepseek-math-7b-instruct",
                    "AI-MO/NuminaMath-7B-CoT",
                    "microsoft/Phi-3-mini-4k-instruct",
                    "microsoft/Phi-3-medium-4k-instruct",
                ]:
                    messages = [{"role": "user", "content": prompt}]
                    print(messages)
                    input_tensor = self.tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True, return_tensors="pt"
                    )
                    outputs = self.model.generate(
                        input_tensor.to(self.model.device),
                        **gen_kwargs,
                    )
                    pred = self.tokenizer.decode(
                        outputs[0][input_tensor.shape[1] :], skip_special_tokens=True
                    )
                else:
                    model_inputs = self.tokenizer(prompt, return_tensors="pt").to(
                        self.model.device
                    )
                    generated_ids = self.model.generate(
                        **model_inputs,
                        **gen_kwargs,
                    )
                    pred = self.tokenizer.batch_decode(
                        generated_ids[:, model_inputs["input_ids"].shape[1] :],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    )[0]
        except Exception as e:
            print(e)
            pred = ""
        return pred



if __name__ == "__main__":
    model = CasualLM("deepseek-ai/deepseek-math-7b-instruct")
    print(model.query("what is your name?"))
    print("DONE")
