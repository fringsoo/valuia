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


"""Classes to use in a basic generative agent.

Based on:

Park, J.S., O'Brien, J.C., Cai, C.J., Morris, M.R., Liang, P. and
Bernstein, M.S., 2023. Generative agents: Interactive simulacra of human
behavior. arXiv preprint arXiv:2304.03442.
"""
from collections.abc import Callable, Sequence
import concurrent
import contextlib
import copy
import datetime
import threading

from concordia.associative_memory import associative_memory
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import agent
from concordia.typing import clock as game_clock
from concordia.typing import component
from concordia.utils import helper_functions
from IPython import display
import termcolor

from concordia.utils import interaction_collector


class BasicAgent(
    agent.GenerativeAgent,
    agent.SpeakerGenerativeAgent,
):
  """A Generative agent."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      memory: associative_memory.AssociativeMemory,
      agent_name: str,
      clock: game_clock.GameClock,
      components: Sequence[component.Component] | None = None,
      num_memories_retrieved: int = 10,
      update_interval: datetime.timedelta = datetime.timedelta(hours=1),
      verbose: bool = False,
      collect_data: bool = False,
      log_color: str = 'green',
      user_controlled: bool = False,
      print_colour='green',
  ):
    """A generative agent.

    Args:
      model: a language model
      memory: an associative memory
      agent_name: the name of the agent
      clock: the game clock is needed to know when is the current time
      components: components that contextualise the policies. The components
        state will be added to the agents state in the order they are passed
        here.
      num_memories_retrieved: number of memories to retrieve for acting,
        speaking, testing
      update_interval: how often to update components. In game time according to
        the clock argument.
      verbose: whether to print chains of thought or not
      user_controlled: if True, would query user input for speech and action
      print_colour: which colour to use for printing
    """
    self._verbose = verbose
    self._print_colour = print_colour

    self._model = model
    self._memory = memory

    self._agent_name = agent_name
    self._clock = clock
    self._num_memories_retrieved = num_memories_retrieved
    self._user_controlled = user_controlled
    self._update_interval = update_interval

    self._under_interrogation = False
    self._state_lock = threading.Lock()
    self._state: str | None

    self._components = {}
    for comp in components:
      self.add_component(comp)

    self._log = []
    self._last_chain_of_thought = None
    self._last_update = datetime.datetime.min
    self._update()


    self._collect_data = collect_data
    if self._collect_data:
        self.interaction_collector = interaction_collector.InteractionCollector(agent_name, "interaction_database")

  @property
  def name(self) -> str:
    return self._agent_name

  def copy(self) -> 'BasicAgent':
    """Creates a copy of the agent."""
    new_sim = BasicAgent(
        model=self._model,
        memory=self._memory,
        agent_name=self._agent_name,
        clock=self._clock,
        components=copy.copy(list(self._components.values())),
        num_memories_retrieved=self._num_memories_retrieved,
        verbose=self._verbose,
        collect_data=self._collect_data,
        user_controlled=self._user_controlled,
        print_colour=self._print_colour,
    )
    return new_sim

  def get_memory(self) -> associative_memory.AssociativeMemory:
    return self._memory

  def _print(self, entry: str):
    print(termcolor.colored(entry, self._print_colour), end='')

  def add_component(self, comp: component.Component) -> None:
    """Add a component."""
    if comp.name() in self._components:
      raise ValueError(f'Duplicate component name: {comp.name()}')
    else:
      self._components[comp.name()] = comp

  def remove_component(self, component_name: str) -> None:
    """Remove a component."""
    del self._components[component_name]

  def set_clock(self, clock: game_clock.GameClock):
    self._clock = clock

  def enter_interrogation(self):
    self._under_interrogation = True

  def leave_interrogation(self):
    self._under_interrogation = False

  @contextlib.contextmanager
  def interrogate(self):
    """Context manager to interrogate the agent.

    When in this context, agent makes no memories or observations and doesn't
    update components.

    Yields:
      None
    """
    self.enter_interrogation()
    try:
      yield
    finally:
      self.leave_interrogation()

  def _ask_for_input(self, context: str, prompt: str) -> str:
    display.clear_output()
    print(context, flush=True)
    result = input(prompt)
    return result

  def get_last_log(self):
    if not self._log:
      return ''
    return self._log[-1]

  def state(self):
    with self._state_lock:
      return '\n'.join(
          f"{self._agent_name}'s " + (comp.name() + ':\n' + comp.state() + '\n')
          for comp in self._components.values()
          if comp.state()
      )

  def _maybe_update(self):
    next_update = self._last_update + self._update_interval
    if self._clock.now() >= next_update and not self._under_interrogation:
      self._update()

  def _update(self):
    self._last_update = self._clock.now()

    def _get_recursive_update_func(
        comp: component.Component,
    ) -> Callable[[], None]:
      return lambda: helper_functions.apply_recursively(
          comp, function_name='update'
      )

    with concurrent.futures.ThreadPoolExecutor() as executor:
      for comp in self._components.values():
        executor.submit(_get_recursive_update_func(comp))

  def observe(self, observation: str):
    if observation and not self._under_interrogation:
      for comp in self._components.values():
        comp.observe(observation)

  def act(
      self,
      action_spec: agent.ActionSpec = agent.DEFAULT_ACTION_SPEC,
      memorize: bool = False,
  ):
    if not action_spec:
      action_spec = agent.DEFAULT_ACTION_SPEC
    self._maybe_update()
    prompt = interactive_document.InteractiveDocument(self._model)
    context_of_action = '\n'.join([
        f'{self.state()}',
    ])

    prompt.statement(context_of_action)

    call_to_action = action_spec.call_to_action.format(
        agent_name=self._agent_name,
        timedelta=helper_functions.timedelta_to_readable_str(
            self._clock.get_step_size()
        ),
    )
    output = ''

    if action_spec.output_type == 'FREE':
      if self._user_controlled:
        output = self._ask_for_input(
            context_of_action,
            call_to_action + '\n',
        )
      else:
        output = self._agent_name + ' '
        answer_to_open_question= prompt.open_question(
            call_to_action,
            max_characters=2500,
            max_tokens=2200,
            answer_prefix=output,
            collect_model_data=self._collect_data,
        )
        if self._collect_data:  
            output += answer_to_open_question[0]
            self.interaction_collector.save_new(answer_to_open_question[1], answer_to_open_question[2])
        else:
            output += answer_to_open_question
    elif action_spec.output_type == 'CHOICE':
      idx = prompt.multiple_choice_question(
          question=call_to_action, answers=action_spec.options
      )
      output = action_spec.options[idx]
    elif action_spec.output_type == 'FLOAT':
      raise NotImplementedError

    def get_externality(externality):
      return externality.update_after_event(output)

    with concurrent.futures.ThreadPoolExecutor() as executor:
      executor.map(get_externality, self._components.values())

    self._last_chain_of_thought = prompt.view().text().splitlines()
    current_log = {
        'date': self._clock.now(),
        'Action prompt': self._last_chain_of_thought,
    }

    for comp in self._components.values():
      last_log = comp.get_last_log()
      if last_log:
        if 'date' in last_log.keys():
          last_log.pop('date')
        current_log[comp.name()] = last_log

    self._log.append(current_log)

    if self._verbose:
      self._print(
          f'\n{self._agent_name} context of action:\n'
          + prompt.view().text()
          + '\n'
      )

    if memorize and not self._under_interrogation:  # observe instead?
      if action_spec.tag:
        self._memory.add(
            f'[{action_spec.tag}] {output}', tags=[action_spec.tag]
        )
      else:
        self._memory.add(output)

    return output

  def add_memory(self, memory: str, importance: float | None = None):
    self._memory.add(memory, importance=importance)

  def say(self, conversation: str) -> str:
    convo_context = (
        f'{self._agent_name} is in the following'
        f' conversation:\n{conversation}\n'
    )
    call_to_speech = agent.DEFAULT_CALL_TO_SPEECH.format(
        agent_name=self._agent_name,
    )
    if self._user_controlled:
      utterance = self._ask_for_input(
          convo_context + call_to_speech, f'{self._agent_name}:'
      )
    else:
      utterance = self.act(
          action_spec=agent.ActionSpec(convo_context + call_to_speech, 'FREE'),
      )

    return utterance

  def train(self):
    raise NotImplementedError
    self._reward_model = None

  def rate(self, instances_ids):
    raise NotImplementedError
    model_ratings = {}
    for iid in instances_ids:
      instance = self.interaction_collector.get_instance(iid)
      model_rating = self._reward_model(instance)
      model_ratings[iid] = model_rating
  
  def upload_ratings(self):
    raise NotImplementedError
    pass

  def download_ratings(self):
    raise NotImplementedError
    pass