import random
import shutil


OPERATIONS = ['+', '-', '*', '/', '%', '^', '|', '&']
OPERANDS = ['x', 'y', 'z', 'm', 'n', 'k', 'p', 'q']
PADDING_TOKEN = 'EOS'


def generate_synthetic_data(num_data=100, max_sequence_len=1, num_operations=4, num_operands=1, early_stop=False):
    """Generates synthetic data sequence using simple context-grammar.

    Note:
        - Each sequence obeys the grammar rule: S -> operand || S operator S
        - Tokens are separated by space

    Args:
        num_data: The number of data sequence to generate.
        max_sequence_len: The maximum number of tokens for each sequence.
        num_operations: The number of operations that will be used in data generation.
        num_operands: The number of operands that will be used in data generation.
        early_stop: If true, the sequence length might be less than max_sequence_len; And the short sequence will be
            padded with the special token 'EOS'.
    """
    assert num_data > 0, 'num_data should be positive'
    assert max_sequence_len >= 1, 'max_sequence_len should be >= 1'
    assert num_operations > 0, 'num_operations should be positive'
    assert num_operations < len(OPERATIONS), 'num_operations should be <= %d' % len(OPERATIONS)
    assert num_operands > 0, 'num_operands should be positive'
    assert num_operands < len(OPERANDS), 'num_operands should be <= %d' % len(OPERANDS)

    if (max_sequence_len - 1) % 2 != 0:
        early_stop = True
        print('Turn on early_stop for generating grammarly correct sequence')

    operations = OPERATIONS[:num_operations] + [PADDING_TOKEN] if early_stop else OPERATIONS[:num_operations]
    operands = OPERANDS[:num_operands]

    file_name = 'synthetic_dataset_%d_data_%d_maxlen_%d_operations_%d_operands' % (
    num_data, max_sequence_len, num_operations, num_operands)
    if early_stop:
        file_name += '_with_earlystop'

    with open(file_name + '.txt', 'w+') as f:
        for line in range(num_data):
            sequence = random.choice(operands) + ' '
            for i in range(1, max_sequence_len, 2):
                next_op = random.choice(operations)
                if next_op == PADDING_TOKEN or i == max_sequence_len - 1:
                    sequence += PADDING_TOKEN
                    break
                sequence = sequence + next_op + ' ' + random.choice(operands) + ' '
            f.write(sequence if line == num_data - 1 else sequence + '\n')

    testdata_file_name = 'testdata/' + file_name + '_test'
    shutil.copy(file_name + '.txt', testdata_file_name + '.txt')


def main():
    generate_synthetic_data(num_data=1000, max_sequence_len=3, num_operations=4, num_operands=1, early_stop=False)
    generate_synthetic_data(num_data=1000, max_sequence_len=15, num_operations=4, num_operands=1, early_stop=False)
    generate_synthetic_data(num_data=1000, max_sequence_len=31, num_operations=4, num_operands=2, early_stop=False)

if __name__ == "__main__":
    main()
