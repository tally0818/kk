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
        super().__init__(model_path=model_path)

    def load_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        if self.use_vllm:
            if self.lora_path is not None:
                raise ValueError(
                    "LoRA loading is only supported in non-vLLM mode in CasualLM. "
                    "Set --use_vllm off or pass a merged checkpoint."
                )
            from vllm import LLM

            self.model = LLM(
                model=model_path,
                tokenizer=model_path,
                gpu_memory_utilization=0.9,
            )

            self.tokenizer = AutoTokenizer.from_pretrained(self.arch)
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

        if self.lora_path is None:
            print(
                f"> Loading the provided {self.arch} checkpoint from '{model_path}'."
            )
        else:
            print(
                f"> Loading base model '{model_path}' with LoRA adapter '{self.lora_path}' "
                f"(tokenizer='{tokenizer_source}')."
            )

    def query(self, prompt):
        return self.query_generation(prompt)
    
    @torch.no_grad()
    def query_generation(self, prompt):
        try:
            if self.use_vllm:
                from vllm import SamplingParams

                sampling_params = SamplingParams(max_tokens=self.max_tokens)
                outputs = self.model.generate(
                    [prompt], sampling_params,
                )
                pred = outputs[0].outputs[0].text
            else:
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
                        max_new_tokens=self.max_tokens,
                    )
                    pred = self.tokenizer.decode(
                        outputs[0][input_tensor.shape[1] :], skip_special_tokens=True
                    )
                else:
                    model_inputs = self.tokenizer(prompt, return_tensors="pt").to(
                        self.model.device
                    )
                    generated_ids = self.model.generate(
                        **model_inputs, max_new_tokens=self.max_tokens
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
