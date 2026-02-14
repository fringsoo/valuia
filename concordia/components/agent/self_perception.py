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

    '''
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
    '''


    question = (
        f'Given the memories above, what kind of person is {self._agent_name}?'
        #f'Given the memories above, what is the terminal value of {self._agent_name}?  Answer in one sentence. Do not be too long.\
        #  Note that it is normal people do not know what they really want at the beginning. Their temporary values might be unconscious polices of their terminal ones. With the change of environment, temporary values might change and people gradually realize what kind of stable values they always want. It is hard to find a deeper motivation for those kind of terminal values. For example, "have a good relationship with others" is not as terminal as "survival", since the former one might be an instinct from long-term evolving under the pressure of later. Of course, there could be more than one terminal value a single people can have.'
    )
    old_state = self._state
    self._state = prompt.open_question(
        question,
        answer_prefix=f'{self._agent_name} is ',
        #answer_prefix=f'The terminal value of {self._agent_name} is ',
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





    '''
    
    questions = [
      "Less powerful members of organizations and institutions rule and power should be distributed unequally.",
      "People in a society should not be integrated into groups.",
      "I am programmed by culture to feel either uncomfortable in unstructured situations.",
      "Different genders should play clearly distinct social and emotional gender roles in a society.",
      "I value reciprocating social obligations, respect for tradition, protecting one's face, and personal steadiness and stability more.",
      "A society should allow relatively free gratification of basic and natural human desires related to enjoying life and having fun.",
      "There is corruption in my country.",
      "I agree that competition is good.",
      "Government monitoring all emails and any other information exchanged on the internet should be legal.",
      "I agree that the government should let anyone from other countries who wants to.",
      "Having the army rule is very good.",
      "Attending peaceful demonstrations is something I have always done.",
      "I completely agree that because of science and technology, there will be more opportunities for the next generation.",
      "I am an active member of women's group.",


      # "One can be a good manager without having a precise answer to every question that a subordinate may raise about his or her work.",
      # "Having pleasant people to work with is important.",
      # "Having a job respected by your family and friends is important.",
      # "I feel proud to be a citizen of my country.",
      # "In my experience, subordinates are never afraid to contradict their boss (or students their teacher)",
      # "All in all, I would describe the state of my health these days as good.",
    ]

    value_dimensions = [
        ["I_AUTHORITY", "Authority orientation measures the extent to which individuals value and respect authority figures and hierarchical structures.", "How important is it for children to learn to obey authority?"],
        ["I_NATIONALISM", "Nationalism indicates the degree of pride and support individuals have for their nation.", "How proud are you of your nationality?"],
        ["I_DEVOUT", "Devoutness measures the level of religious commitment and importance of religion in an individual's life.", "How important is religion in your daily life?"],
        ["DEFIANCE", "Defiance reflects a resistance to authority and traditional norms.", "Do you think it is important to challenge and question traditional values and authorities?"],
        ["I_RELIGIMP", "Importance of religion indicates how central religion is to an individual's worldview and daily activities.", "How often do you attend religious services?"],
        ["I_RELIGBEL", "Religious belief measures the strength of personal belief in religious teachings and doctrines.", "How strongly do you believe in the existence of God or a higher power?"],
        ["I_RELIGPRAC", "Religious practice assesses the extent to which individuals engage in religious rituals and practices.", "How often do you pray or meditate?"],
        ["DISBELIEF", "Disbelief measures skepticism or rejection of religious beliefs.", "Do you consider yourself an atheist?"],
        ["I_NORM1", "Norm1 typically refers to societal norms and values related to behavior and ethics.", "How important is it to follow societal norms and customs?"],
        ["I_NORM2", "Norm2 typically refers to societal norms and values related to behavior and ethics.", "Do you believe in maintaining traditional family values?"],
        ["I_NORM3", "Norm3 typically refers to societal norms and values related to behavior and ethics.", "How important is it to adhere to cultural norms in your community?"],
        ["RELATIVISM", "Relativism assesses the belief that moral principles are not absolute and can vary based on context.", "Do you agree that what is right or wrong depends on the situation?"],
        ["I_TRUSTARMY", "Trust in the army measures the confidence individuals have in the military.", "How much do you trust the army?"],
        ["I_TRUSTPOLICE", "Trust in the police assesses the confidence in law enforcement agencies.", "How much do you trust the police?"],
        ["I_TRUSTCOURTS", "Trust in the courts measures confidence in the judicial system.", "How much do you trust the legal system and courts?"],
        ["SCEPTICISM", "Scepticism reflects a questioning attitude towards accepted beliefs or claims.", "How often do you question information presented by the media or authorities?"],
        ["SACSECVAL", "Sacred vs. Secular Values measures the importance of sacred/religious values compared to secular values.", "Do you believe religious values should influence laws and government policies?"],
        ["I_INDEP", "Independence measures the value placed on personal autonomy and self-direction.", "How important is it for you to be self-reliant and independent?"],
        ["I_IMAGIN", "Imagination assesses the importance placed on creativity and imaginative thinking.", "How important is it to encourage creativity and imagination in children?"],
        ["I_NONOBED", "Non-obedience measures the value placed on questioning and challenging authority rather than conforming.", "Do you believe it is important to teach children to question authority?"],
        ["AUTONOMY", "Autonomy reflects the value placed on individual freedom and self-governance.", "How important is personal freedom and independence to you?"],
        ["I_WOMJOB", "Women in the workforce measures attitudes towards women's participation in the labor market.", "Do you support equal job opportunities for men and women?"],
        ["I_WOMPOL", "Women in politics assesses attitudes towards women's participation in political life.", "Do you support the idea of women holding political office?"],
        ["I_WOMEDU", "Women in education measures support for equal educational opportunities for women.", "Do you believe that women should have the same educational opportunities as men?"],
        ["EQUALITY", "Equality reflects the value placed on equal rights and opportunities for all people.", "How important is it to you that everyone is treated equally, regardless of their background?"],
        ["I_HOMOLIB", "Homosexuality liberation measures attitudes towards the acceptance and rights of LGBTQ+ individuals.", "Do you support equal rights for LGBTQ+ individuals?"],
        ["I_ABORTLIB", "Abortion liberation measures attitudes towards the right to choose an abortion.", "Do you believe women should have the right to choose an abortion?"],
        ["I_DIVORLIB", "Divorce liberation assesses attitudes towards the acceptability and rights to divorce.", "Do you believe that individuals should have the right to divorce if they are unhappy in their marriage?"],
        ["CHOICE", "Choice measures the value placed on individual choice and personal decision-making.", "How important is it to you to have the freedom to make your own choices in life?"],
        ["RESEMAVAL", "Resemanticization of values measures how values are reinterpreted or given new meanings in different contexts.", "Do you think societal values should be reinterpreted to fit modern times?"]
        ]
    questions = [vd[2]  for vd in value_dimensions]

    answers = []
    for q in questions:
      prompt = interactive_document.InteractiveDocument(self._model)
      prompt.statement(f'Memories of {self._agent_name}:\n{mems}')

      if self._clock_now is not None:
        prompt.statement(f'Current time: {self._clock_now()}.\n')

      question = (f'This is a cultural value survey. Does {self._agent_name} agree with the opinion that "{q}"  Just respond with a real number between 0 and 1, representing the extent of your agreement to the question.  No other notation marks or explanations.')
      while True:
        try:
          value = prompt.open_question(
            question,
            #answer_prefix=f'{self._agent_name} is ',
            answer_prefix=f'The answer is: ',
            )
          #assert value in ("yes", "no")
          value = float(value)
          assert value >= 0 and value <= 1
          answers.append(value)
          break
        except:
          pass
    self._answers = dict(zip([vd[0] for vd in value_dimensions], answers))
    '''
    #########################################################################
    import pandas as pd
    self._answers = {}
    df_valuebench = pd.read_csv('../../model_data/value_data/value_orientation.csv')
    grouped = df_valuebench.groupby('value')

    with open('memories.txt', 'a') as f:
      f.write(f'Memories of {self._agent_name}:\n{mems}')
      f.write('\n#########################################\n')
      
    value_count = 0
    for value_name, group in grouped:
      if value_count > 100:
        break
      print(f"valuecount{value_count}")
      value_count += 1
      prompt = interactive_document.InteractiveDocument(self._model)
      prompt.statement(f'Memories of {self._agent_name}:\n{mems}')

      if self._clock_now is not None:
        prompt.statement(f'Current time: {self._clock_now()}.\n')

      answers = []
      for groupmember in zip(group['item'], group['agreement']):
        q = groupmember[0]
        a = groupmember[1]

        question = (f'Will {self._agent_name} agree with the following sentence as its self-statement? "{q}"  Just respond with a real number between -1 and 1, -1 representing totally disagreement and 1 representing totally agreement. No other notation marks, comma, full stop or explanations.')
        while True:
          try:
            value = prompt.open_question(
              question,
              #answer_prefix=f'{self._agent_name} is ',
              answer_prefix=f'The answer is: ',
              )
            #assert value in ("yes", "no")
            value = float(value)
            assert value >= -1 and value <= 1
            value = value * a
            answers.append(value)
            break
          except:
            pass
      answer = sum(answers) / len(answers)
      answers_std = sum([(a - answer)**2 for a in answers]) / len(answers)
      self._answers[value_name] = answer
      print(value_name, answer, answers_std)
    #self._answers = dict(zip([vd[0] for vd in value_dimensions], answers))
