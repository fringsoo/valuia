"""Local Llama3 Language Model."""

from collections.abc import Collection, Sequence
import re

from concordia.language_model import language_model
from concordia.utils import measurements as measurements_lib
from langchain import llms
from typing_extensions import override


def _extract_choices(text):
  match = re.search(r"\(?(\w)\)", text)
  if match:
    return match.group(1)
  return None


class Llama3LanguageModel(language_model.LanguageModel):
  """Language Model that uses Ollama LLM models."""

  def __init__(
      self,
      model_name: str,
      *,
      system_message: str = "",
      measurements: measurements_lib.Measurements | None = None,
      channel: str = language_model.DEFAULT_STATS_CHANNEL,
  ) -> None:
    """Initializes the instance.

    Args:
        model_name: The language model to use. For more details, see
          https://github.com/ollama/ollama.
        system_message: System message to prefix to requests when prompting the
          model.
        measurements: The measurements object to log usage statistics to.
        channel: The channel to write the statistics to.
    """
    self._model_name = model_name
    self._system_message = system_message
    self._measurements = measurements
    self._channel = channel

    from langchain import HuggingFacePipeline
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    from transformers import BitsAndBytesConfig
    import torch
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    #bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto", low_cpu_mem_usage=True) 
    #model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True)
    model.resize_token_embeddings(len(tokenizer))
    self._client = pipeline("text-generation", model=model, tokenizer=tokenizer, 
                            # model_kwargs={
                            #   "torch_dtype": torch.float16,
                            #   "quantization_config": {"load_in_4bit": True},
                            #   "low_cpu_mem_usage": True,
                            #   },
                              )
    #self._client  = HuggingFacePipeline(pipeline=self._client)


  @override
  def sample_text(
      self,
      prompt: str,
      *,
      max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
      max_characters: int = language_model.DEFAULT_MAX_CHARACTERS,
      terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
      temperature: float = language_model.DEFAULT_TEMPERATURE,
      timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
      seed: int | None = None,
  ) -> str:
    messages = [
    {"role": "system", "content": self._system_message},
    {"role": "user", "content": prompt},
]

    prompt_chat_template = self._client.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
    )

    terminators = [
        self._client.tokenizer.eos_token_id,
        self._client.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = self._client(
        prompt_chat_template,
        max_new_tokens=max_tokens,
        eos_token_id=terminators,
        pad_token_id=self._client.tokenizer.eos_token_id,
        temperature=temperature,
        top_p=0.9,
    )

    return outputs[0]["generated_text"][len(prompt_chat_template):]

  @override
  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
  ) -> tuple[int, str, dict[str, float]]:
    max_characters = len(max(responses, key=len))
    #prompt_with_system_message = f"{self._system_message}\n\n{prompt}"
    sample = self.sample_text(
        #prompt_with_system_message,
        prompt + '\nJust choose one of the options. No more talk.',
        max_characters=max_characters,
        temperature=0.1,
        seed=seed,
    )
    answer = _extract_choices(sample)
    try:
      idx = responses.index(answer)
    except ValueError:
      raise language_model.InvalidResponseError(
          f"Invalid response: {answer}. "
          f"Valid responses: {responses}"
          f"LLM Input: {prompt}\nLLM Output: {sample}"
      ) from None

    if self._measurements is not None:
      self._measurements.publish_datum(self._channel, {"choices_calls": 1})
    debug = {}
    return idx, responses[idx], debug
