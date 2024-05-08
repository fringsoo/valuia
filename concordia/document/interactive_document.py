# Copyright 2023 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Utilities for chain-of-thought prompting."""

from collections.abc import Collection, Iterable, Iterator, Sequence
import contextlib

from concordia.document import document
from concordia.language_model import language_model
import numpy as np

DEFAULT_MAX_CHARACTERS = 200
DEFAULT_MAX_TOKENS = DEFAULT_MAX_CHARACTERS // 4

DEBUG_TAG = 'debug'
STATEMENT_TAG = 'statement'
QUESTION_TAG = 'question'
RESPONSE_TAG = 'response'
MODEL_TAG = 'model'
INTERACTIVE_TAGS = frozenset(
    {DEBUG_TAG, STATEMENT_TAG, QUESTION_TAG, RESPONSE_TAG, MODEL_TAG}
)


_YESNO = ['No', 'Yes']


def _letters():
  """Yields the letters from a to z."""
  yield from (chr(ord('a') + i) for i in range(26))


class InteractiveDocument(document.Document):
  """A document formed by interaction with a language model."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      contents: Iterable[document.Content] = (),
      rng: np.random.Generator | None = None,
  ) -> None:
    """Initializes the instance.

    Args:
      model: language model to interact with.
      contents: initial contents of the document.
      rng: randomization source.
    """
    super().__init__(contents)
    if rng:
      self._rng = rng
    else:
      self._rng = np.random.default_rng()
    self._model = model
    self._model_view = self.view()
    # TODO: b/311191701 - debug log some useful stuff?

  def view(
      self,
      include_tags: Iterable[str] = (),
      exclude_tags: Iterable[str] = (DEBUG_TAG,),
  ) -> document.View:
    """Returns a view of the document.

    Args:
      include_tags: specifies which tags to include in the view.
      exclude_tags: specifies which tags to exclude from the view.
    """
    return super().view(include_tags=include_tags, exclude_tags=exclude_tags)

  def copy(self) -> 'InteractiveDocument':
    """See base class."""
    # TODO: b/311192069 - what about rng?
    return InteractiveDocument(
        model=self._model, contents=self.contents(), rng=self._rng
    )

  @contextlib.contextmanager
  def edit(self) -> Iterator['InteractiveDocument']:
    """See base class."""
    # TODO: b/311192069 - what about rng?
    edit = InteractiveDocument(model=self._model, rng=self._rng)
    yield edit
    self.extend(edit.contents())

  def debug(
      self, text: str, *, tags: Collection[str] = (), end: str = '\n'
  ) -> None:
    """Appends debug text to the document.

    Args:
      text: text to append.
      tags: additional tags for appended text.
      end: appended to `text`.
    """
    self.append(text + end, tags=[DEBUG_TAG, *tags])

  def statement(
      self, text: str, *, tags: Collection[str] = (), end: str = '\n'
  ) -> None:
    """Appends a statement to the document.

    Args:
      text: text to append.
      tags: additional tags for appended text.
      end: appended to `text`.
    """
    self.append(text + end, tags=[STATEMENT_TAG, *tags])

  def _question(
      self, text: str, *, tags: Collection[str] = (), end: str = ''
  ) -> None:
    """Appends a question to the document."""
    self.append(text + end, tags=[QUESTION_TAG, *tags])

  def _response(
      self, text: str, *, tags: Collection[str] = (), end: str = ''
  ) -> None:
    """Appends a response to the document."""
    self.append(text + end, tags=[RESPONSE_TAG, *tags])

  def _model_response(
      self, text: str, *, tags: Collection[str] = (), end: str = ''
  ) -> None:
    """Appends a response to the document that was generated by the model."""
    self.append(text + end, tags=[RESPONSE_TAG, MODEL_TAG, *tags])

  def open_question(
      self,
      question: str,
      *,
      forced_response: str | None = None,
      answer_prefix: str = '',
      answer_suffix: str = '',
      max_tokens: int = DEFAULT_MAX_TOKENS,
      max_characters: int = DEFAULT_MAX_CHARACTERS,
      terminators: Collection[str] = ('\n',),
      collect_model_data: bool = False,
  ) -> str:
    """Asks the agent an open question and appends it to the document.

    Args:
      question: the question to ask.
      forced_response: forces the document to provide this response. The LLM
        will not be consulted. If answer_prefix is in the forced response then
        remove it.
      answer_prefix: a prefix to append to the model's prompt.
      answer_suffix: a suffix to append to the model's response.
      max_tokens: the maximum number of tokens to sample from the model.
      max_characters: the maximum number of characters to sample from the model.
      terminators: strings that must not be present in the model's response. If
        emitted by the model the response will be truncated before them.

    Returns:
      The agents truncated response (or `forced_response` is provided).
    """
    self._question(f'Question: {question}\n')
    self._response(f'Answer: {answer_prefix}')
    if forced_response is None:
    
      if not collect_model_data:
        response = self._model.sample_text(
          prompt=self._model_view.text(),
          max_tokens=max_tokens,
          max_characters=max_characters,
          terminators=terminators,
          )
      else:
        sample_no = 3
        agent_fine_tuning_data_collect_samples = [] 
        scenario_text =  self._model_view.text()
        for rr in range(sample_no):
          response = self._model.sample_text(
              prompt=scenario_text,
              max_tokens=max_tokens,
              max_characters=max_characters,
              terminators=terminators,
              temperature=1,
          )
          agent_fine_tuning_data_collect_samples.append(response)  
    else:
      response = forced_response
    response = response.removeprefix(answer_prefix)
    self._model_response(response)
    self._response(f'{answer_suffix}\n')

    try:
      return response, scenario_text, agent_fine_tuning_data_collect_samples
    except Exception as e:
      return response

  def multiple_choice_question(
      self, question: str, answers: Sequence[str]
  ) -> int:
    """Presents a multiple choice to the agent.

    Args:
      question: the question to ask the agent.
      answers: the choice of answers

    Returns:
      The index of the sampled answer.
    """
    original_indices = self._rng.permutation(len(answers))
    options = {key: answers[i] for key, i in zip(_letters(), original_indices)}
    self._question(f'Question: {question}\n')
    for key, option in options.items():
      self._question(f'  ({key}) {option}\n')

    self._response('Answer: (')
    idx, response, debug = self._model.sample_choice(
        prompt=self._model_view.text(),
        responses=list(options.keys()),
    )
    self._model_response(response)
    self._response(')\n')
    self.debug(f'[{debug}]')
    return original_indices[idx]

  def yes_no_question(self, question: str) -> bool:
    """Presents a yes/no question to the agent.

    Args:
      question: the question to ask the agent.

    Returns:
      True iff the answer was answered with Yes.
    """
    return self.multiple_choice_question(question, _YESNO) == _YESNO.index(
        'Yes'
    )
