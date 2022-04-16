# numbrix.py: Template para implementação do projeto de Inteligência Artificial 2021/2022.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# Grupo 19:
# 96915 Tomás Nunes
# 95597 João Silveira

from difflib import diff_bytes
import sys
from copy import deepcopy

# from matplotlib.pyplot import fill
from search import Problem, Node, astar_search, breadth_first_tree_search, depth_first_tree_search, greedy_search, recursive_best_first_search

def manhattan_distance(x1, x2):
    return sum(abs(val1 - val2) for val1, val2 in zip(x1, x2))

class NumbrixState:
    state_id = 0

    def __init__(self, board):
        self.id = NumbrixState.state_id
        NumbrixState.state_id += 1

        self.board = board

    def __lt__(self, other):
        return self.id < other.id

    def fill(self, row: int, col: int, number: int):
        """
        Preenche um quadrado no tabuleiro
        """
        copy = deepcopy(self)
        copy.board.fill(row, col, number)
        return copy


class Board:
    def __init__(self, n: int, squares: list):
        """
        Representação interna de um tabuleiro de Numbrix.
        """
        self.n = n                                                                                                                # Constant value
        self.n_squares = n * n                                                                                                    # Constant value
        self.n_zeros = sum(map(lambda x: x == 0, [square for row in squares for square in row]))                                  # Number of Zeros on board
        self.squares = squares                                                                                                    # Square Indexed Matrix
                 
        self.numbers = [None] * self.n_squares                                                                                    # Number Indexed Array
        for i in range(self.n):
            for j in range(self.n):
                if squares[i][j] != 0:
                    self.numbers[self.squares[i][j] - 1] = (i, j)

        self.possible_actions = [self.actions_for_number(number + 1) if self.numbers[number] == None else [] for number in range(self.n_squares)]  # Number Indexed Array of possible actions                                                                # Number Indexed Array of Actions

    def adjacent_vertical_numbers(self, row: int, col: int):
        """
        Devolve os valores imediatamente abaixo e acima,
        respectivamente.
        """
        down = self.squares[row + 1][col] if row + 1 < self.n else None
        up = self.squares[row - 1][col] if row - 1 >= 0 else None
        return (down, up)

    def adjacent_horizontal_numbers(self, row: int, col: int):
        """
        Devolve os valores imediatamente à esquerda e à direita,
        respectivamente.
        """
        left = self.squares[row][col - 1] if col - 1 >= 0 else None
        right = self.squares[row][col + 1] if col + 1 < self.n else None
        return (left, right)

    def adjacent_all_numbers(self, row: int, col: int):
        """
        Devolve os valores imediatamente à esquerda, direita,
        cima e baixo respetivamente.
        """
        left, right = self.adjacent_horizontal_numbers(row, col)
        down, up = self.adjacent_vertical_numbers(row, col)
        return (left, right, down, up)

    def fill(self, row : int, col : int, number : int):
        """
        Preenche o tabuleiro com um numero numa dada posicao
        """
        self.n_zeros = self.n_zeros - 1
        self.squares[row][col] = number
        self.numbers[number - 1] = (row, col)

        self.possible_actions[number - 1] = []
        for i in range(self.n_squares):
            self.possible_actions[i] = [action for action in self.possible_actions[i] if action[0] != row or action[1] != col]

        if number != 1 and self.numbers[number - 2] == None:
            self.possible_actions[number - 2] = self.actions_for_number(number - 1)

        if number != self.n_squares and self.numbers[number] == None:
            self.possible_actions[number] = self.actions_for_number(number + 1)

    def actions_for_number(self, number) -> list:
        possible_left = []
        if number != 1 and self.numbers[number - 2] != None:
            x1 = self.numbers[number - 2][0]
            y1 = self.numbers[number - 2][1]

            left, right, down, up = self.adjacent_all_numbers(x1, y1)
            if left == 0:
                possible_left.append((x1, y1 - 1, number))
            if right == 0:
                possible_left.append((x1, y1 + 1, number))
            if down == 0:
                possible_left.append((x1 + 1, y1, number))
            if up == 0:
                possible_left.append((x1 - 1, y1, number))

        possible_right = []
        if number != self.n_squares and self.numbers[number] != None:
            x1 = self.numbers[number][0]
            y1 = self.numbers[number][1]

            left, right, down, up = self.adjacent_all_numbers(x1, y1)
            if left == 0:
                possible_right.append((x1, y1 - 1, number))
            if right == 0:
                possible_right.append((x1, y1 + 1, number))
            if down == 0:
                possible_right.append((x1 + 1, y1, number))
            if up == 0:
                possible_right.append((x1 - 1, y1, number))

        intersection = [number for number in possible_left if number in possible_right]
        return intersection if intersection else possible_left + possible_right 

    def to_string(self):
        """
        Print do tabuleiro
        """
        return '\n'.join(['\t'.join([str(number) for number in row]) for row in self.squares])

    @staticmethod
    def parse_instance(filename: str):
        """
        Lê o ficheiro cujo caminho é passado como argumento e retorna
        uma instância da classe Board.
        """
        with open(filename, "r") as f:
            N = int(f.readline())
            board = Board(N, [[int(num) for num in f.readline().split("\t")] for _ in range(N)])
            f.close()
        return board


class Numbrix(Problem):
    def __init__(self, board: Board):
        """
        O construtor especifica o estado inicial.
        """
        self.initial = NumbrixState(board)

    def actions(self, state: NumbrixState) -> list:
        """
        Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento.
        """
        n_minimum_actions = float("inf")
        minimum_actions = []
        for number in range(state.board.n_squares):
            n_actions = len(state.board.possible_actions[number])
            if n_actions and n_actions < n_minimum_actions:
                n_minimum_actions = n_actions
                minimum_actions = state.board.possible_actions[number]
        return minimum_actions

    def result(self, state: NumbrixState, action) -> NumbrixState:
        """
        Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state).
        """
        row, col, number = action
        return state.fill(row, col, number)

    def goal_test(self, state: NumbrixState) -> bool:
        """
        Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas com uma sequência de números adjacentes.
        """
        if state.board.n_zeros != 0:
            return False

        for i in range(state.board.n_squares - 1):
            x1 = state.board.numbers[i][0]
            y1 = state.board.numbers[i][1]
            x2 = state.board.numbers[i + 1][0]
            y2 = state.board.numbers[i + 1][1]
            if manhattan_distance((x1, y1), (x2, y2)) > 1:
                return False

        return True

    def h(self, node: Node):
        """
        Função heuristica utilizada para a procura A*.
        """
        return node.state.board.n_zeros

if __name__ == "__main__":
    board = Board.parse_instance(sys.argv[1])
    problem = Numbrix(board)
    goal_node = greedy_search(problem, problem.h)
    if goal_node != None:
        print(goal_node.state.board.to_string(), sep="")
    else:
        print("Found no solution!")