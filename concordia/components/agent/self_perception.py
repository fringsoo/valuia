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

"""Agent component for self perception."""
import datetime
from typing import Callable

from concordia.associative_memory import associative_memory
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import component
import termcolor


class SelfPerception(component.Component):
  """This component answers the question 'what kind of person is the agent?'."""

  def __init__(
      self,
      name: str,
      model: language_model.LanguageModel,
      memory: associative_memory.AssociativeMemory,
      agent_name: str,
      clock_now: Callable[[], datetime.datetime] | None = None,
      num_memories_to_retrieve: int = 100,
      verbose: bool = False,
  ):
    """Initializes the SelfPerception component.

    Args:
      name: Name of the component.
      model: Language model.
      memory: Associative memory.
      agent_name: Name of the agent.
      clock_now: time callback to use for the state.
      num_memories_to_retrieve: Number of memories to retrieve.
      verbose: Whether to print the state.
    """

    self._verbose = verbose
    self._model = model
    self._memory = memory
    self._state = ''
    self._agent_name = agent_name
    self._clock_now = clock_now
    self._num_memories_to_retrieve = num_memories_to_retrieve
    self._name = name
    self._last_update = self._clock_now() - datetime.timedelta(days=365)
    self._history = []
    self._value_bases = ['satisfying curiosity', 'physical energy and comfort', 'feeling safety', 'mental happiness', 'intimacy with others', 'fairness for self and others']
    self._value_weights = None

  def name(self) -> str:
    return self._name

  def state(self) -> str:
    return self._state

  def get_last_log(self):
    if self._history:
      return self._history[-1].copy()

  def update(self) -> None:
    if self._clock_now() == self._last_update:
      return
    self._last_update = self._clock_now()

    mems = '\n'.join(
        self._memory.retrieve_recent(
            self._num_memories_to_retrieve, add_time=True
        )
    )

    prompt = interactive_document.InteractiveDocument(self._model)
    prompt.statement(f'Memories of {self._agent_name}:\n{mems}')

    if self._clock_now is not None:
      prompt.statement(f'Current time: {self._clock_now()}.\n')

    sample_piece_no = len(self._value_bases) + 1
    question = (f'Given the memories above, sample {sample_piece_no} piece(s) of memories from them. Leave your throughout score (a real number between 0 and 1, 0 representing bad and 1 representing good) of each of the piece by instinct.\nThe format is:\n1. <The first piece of memory>, <the score by instinct>.\n2. <The second piece of memory>, <the score by instinct>.\n3. <The third piece of memory>, <the score by instinct>.\n...\n Only output put the pieces of memories and their scores strictly following the format, do not output any notations or explanations.')

    while True:
      try:
        memory_scores = prompt.open_question(
        question,
        #answer_prefix=f'{self._agent_name} is ',
        answer_prefix=f'The samples and their scores are: ',
        max_characters=3000,
        max_tokens=1000,
        )

        mm_samples = []
        score_samples = []
        memory_pieces = memory_scores.splitlines()
        assert len(memory_pieces) == sample_piece_no
        for p in memory_pieces:
          mm, score = p.split('.,')
          mm_samples.append(mm[2:])
          score = float(score)
          assert score >= 0 and score <= 1
          score_samples.append(score)
        break
      except Exception as e:
        print(e)
    
    lines_of_base_scores = []
    
    for mm in mm_samples:
      question = (f'On each of the following value dimensions ({self._value_bases}), how would {self._agent_name} score (using a real number between 0 and 1, 0 representing bad and 1 representing good) this memory piece : {mm}. Only respond with the scores in the order of the value dimensions in one line, splitting with commas, no other text.')
      
      while True:
        try:
          base_scores = prompt.open_question(
            question,
            #answer_prefix=f'{self._agent_name} is ',
            answer_prefix=f'The scores are: ',
            max_characters=3000,
            max_tokens=1000,
          )
          base_scores = [float(sss) for sss in base_scores.split(',')]
          assert len(base_scores) == len(self._value_bases)
          for sss in base_scores:
            assert sss >= 0 and sss <= 1
          lines_of_base_scores.append(base_scores)
          break
        except Exception as e:
          print(e)

    import numpy as np 
    lines_of_base_scores = np.array(lines_of_base_scores.append([1 for rr in range(len(self._value_bases))]))
    score_samples = np.array(score_samples.append(1))
    weights = np.linalg.solve(lines_of_base_scores[-len(self._value_bases):], score_samples[-len(self._value_bases):])
    print(weights)

    question = (f'Apart from the following value dimensions ({self._value_bases}, add another value dimension that is important.')
    new_dimension = prompt.open_question(
      question,
            #answer_prefix=f'{self._agent_name} is ',
      answer_prefix=f'The new dimension is: ',
      max_characters=3000,
      max_tokens=1000,
      )
    self._value_bases.append(new_dimension)    
    
    # rank = np.linalg.matrix_rank(lines_of_base_scores)    
    # if rank < len(self._value_bases):
    #   pass
    # else: 
    #   diff = np.matmul(weights, lines_of_base_scores) - scores



    question = (
        #f'Given the memories above, what kind of person is {self._agent_name}?'
        f'Given the memories above, what is the terminal value of {self._agent_name}?  Answer in one sentence. Do not be too long.\
          Note that it is normal people do not know what they really want at the beginning. Their temporary values might be unconscious polices of their terminal ones. With the change of environment, temporary values might change and people gradually realize what kind of stable values they always want. It is hard to find a deeper motivation for those kind of terminal values. For example, "have a good relationship with others" is not as terminal as "survival", since the former one might be an instinct from long-term evolving under the pressure of later. Of course, there could be more than one terminal value a single people can have.'
    )
    old_state = self._state
    self._state = prompt.open_question(
        question,
        #answer_prefix=f'{self._agent_name} is ',
        answer_prefix=f'The terminal value of {self._agent_name} is ',
        max_characters=3000,
        max_tokens=1000,
    )

    self._state = f'{self._agent_name} is {self._state}'

    if old_state != self._state:
      self._memory.add(f'[self reflection] {self._state}')

    self._last_chain = prompt
    if self._verbose:
      print(termcolor.colored(self._last_chain.view().text(), 'green'), end='')

    update_log = {
        'date': self._clock_now(),
        'Summary': question,
        'State': self._state,
        'Chain of thought': prompt.view().text().splitlines(),
    }
    self._history.append(update_log)
