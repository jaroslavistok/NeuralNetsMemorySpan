import random


class ReberAutomaton:
    def __init__(self):
        self.states = self._create_states()

    def generate_reber_string(self):
        generated_string = ''
        current_state = self.states[0]
        while True:
            random_transition = random.randint(0, 1)
            if len(current_state) < 2:
                random_transition = 0
            transition = current_state[random_transition]
            for next_state, character in transition.items():
                generated_string += character
                current_state = self.states[next_state]
                if next_state == -1:
                    return generated_string

    def _create_states(self):
        states = [[{1: 'T'}, {2: 'P'}],
                  [{1: 'S'}, {3: 'X'}],
                  [{2: 'T'}, {4: 'V'}],
                  [{5: 'S'}, {2: 'X'}],
                  [{3: 'P'}, {5: 'V'}],
                  [{-1: 'E'}]]
        return states

    def _generate_reber_strings(self, number_of_strings):
        generated_string = ''
        for i in range(number_of_strings):
            generated_string += self.generate_reber_string() + ' '
        return generated_string

    def generate_reber_strings_dataset(self, number_of_strings):
        generated_reber_string = self._generate_reber_strings(number_of_strings)
        with open('data/sequences/reber_strings', 'w') as file:
            file.write(generated_reber_string)
