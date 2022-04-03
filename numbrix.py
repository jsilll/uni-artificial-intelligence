# numbrix.py: Template para implementação do projeto de Inteligência Artificial 2021/2022.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# Grupo 19:
# 96915 Tomás Nunes
# 95597 João Silveira

import sys
from copy import deepcopy

from matplotlib.pyplot import fill
from search import Problem, Node, astar_search, breadth_first_tree_search, depth_first_tree_search, greedy_search, recursive_best_first_search

class NumbrixState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = NumbrixState.state_id
        NumbrixState.state_id += 1

    def __lt__(self, other):
        return self.id < other.id

    def fill(self, row: int, col: int, number: int):
        """
        Preenche um quadrado no tabuleiro
        """

        state_copy = deepcopy(self)
        state_copy.board.squares[row][col] = number
        state_copy.board.numbers[number - 1] = (row, col)
        state_copy.board.zeros = state_copy.board.zeros - 1
        return state_copy


class Board:
    def __init__(self, n: int, squares: list):
        """
        Representação interna de um tabuleiro de Numbrix.
        """

        self.n = n
        self.n_squares = n * n
        self.squares = squares
        self.numbers = [None] * self.n_squares
        self.zeros = sum(map(lambda x: x == 0, [square for row in squares for square in row]))

        for i in range(self.n):
            for j in range(self.n):
                if squares[i][j] != 0:
                    self.numbers[self.squares[i][j] - 1] = (i, j)

    def get_number(self, row: int, col: int) -> int:
        """
        Devolve o valor na respetiva posição do tabuleiro.
        """

        return self.squares[row][col]

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

    def get_neighbors(self, row : int, col : int):
        left, right = self.adjacent_horizontal_numbers(row, col)
        down, up = self.adjacent_vertical_numbers(row, col)
        return (left, right, down, up)

    def to_string(self):
        """
        Print do tabuleiro
        """

        return '\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in self.squares])

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
        res = []
        for i in range(state.board.n):
            for j in range(state.board.n):
                if state.board.squares[i][j] == 0:
                    neighbors = state.board.get_neighbors(i, j)
                    n_valid_neighbors = len([n for n in neighbors if n != None and n != 0])
                    possible_actions_sq = [action for action in self.get_possible_values(i, j, state)]
                    if n_valid_neighbors > 1 and possible_actions_sq == []:
                        return []
                    res += possible_actions_sq
        return res

    def get_possible_values(self, row: int, col: int, state: NumbrixState):
        """
        Returns a list of possible values for a given square.
        This square's value must always satisfy at least 1 adjacent square.
        """

        if state.board.squares[row][col] != 0:
            return

        neighbors = list(state.board.adjacent_vertical_numbers(row, col)) + list(state.board.adjacent_horizontal_numbers(row, col))
        valid_neighbors = list(filter(lambda x: x != None and x != 0, neighbors))

        n_valid_neighbors = len(valid_neighbors)
        n_edges = neighbors.count(None)

        if n_valid_neighbors == 0:
            return

        candidates = {}
        for neighbor in valid_neighbors:
            candidate_upper = neighbor + 1
            candidate_lower = neighbor - 1
            if candidate_upper < state.board.n_squares + 1 and state.board.numbers[candidate_upper - 1] == None:
                candidates[candidate_upper] = candidates.get(candidate_upper, 0) + 1
            if candidate_lower > 0 and state.board.numbers[candidate_lower - 1] == None:
                candidates[candidate_lower] = candidates.get(candidate_lower, 0) + 1

        if 1 in candidates.keys():
            yield (row, col, 1)
            del candidates[1]

        if state.board.n_squares in candidates.keys():
            yield (row, col, state.board.n_squares)
            del candidates[state.board.n_squares]

        if n_edges == 0:
            if n_valid_neighbors <= 3:
                for possible_value in candidates.items():
                    yield (row, col, possible_value[0])
            else:
                for possible_value in filter(lambda x: x[1] == 2, candidates.items()):
                    yield (row, col, possible_value[0])
        elif n_edges == 1:
            if n_valid_neighbors <= 2:
                for possible_value in candidates.items():
                    yield (row, col, possible_value[0])
            else:
                for possible_value in filter(lambda x: x[1] == 2, candidates.items()):
                    yield (row, col, possible_value[0])
        elif n_edges == 2:
            for possible_value in filter(lambda x: x[1] == n_valid_neighbors, candidates.items()):
                yield (row, col, possible_value[0])


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

        if state.board.zeros != 0:
            return False
        for i in range(1, state.board.n_squares):
            x1 = state.board.numbers[i][0]
            y1 = state.board.numbers[i][1]
            x2 = state.board.numbers[i - 1][0]
            y2 = state.board.numbers[i - 1][1]
            man_distance = abs(x1 - x2) + abs(y1 - y2) - 1
            if man_distance > 1:
                return False
        return True

    # TODO: performance can be improved, iterate only on number gaps
    def h(self, node: Node):
        """
        Função heuristica utilizada para a procura A*.
        """

        filled_positions = [x for x in  node.state.board.numbers if x != None]

        for i in range(len(filled_positions) - 1):
            x1 = filled_positions[i][0]
            x2 = filled_positions[i + 1][0]
            y1 = filled_positions[i][1]
            y2 = filled_positions[i + 1][1]

            num_distance = abs(node.state.board.squares[x1][y1] - node.state.board.squares[x2][y2])
            if num_distance == 1:
                continue

            man_distance = abs(x1 - x2) + abs(y1 - y2)
            if man_distance > num_distance:
                return float("inf")

        for i in range(len(filled_positions)):
            x = filled_positions[i][0]
            y = filled_positions[i][1]

            n = node.state.board.squares[x][y]
            neighbors = node.state.board.get_neighbors(x, y)

            n_zeros_neighbors = len([neigh for neigh in neighbors if neigh == 0])
            n_valid_neighbors = len([neigh for neigh in neighbors if neigh != 0 and neigh != None and abs(neigh - n) == 1])

            if n == 1 or n == node.state.board.n_squares:
                if n_zeros_neighbors == 0 and n_valid_neighbors != 1:
                    return float("inf")
            else:
                if n_zeros_neighbors == 1 and n_valid_neighbors < 1:
                    return float("inf")
                elif n_zeros_neighbors == 0 and n_valid_neighbors < 2:
                    return float("inf")

        return node.state.board.zeros


if __name__ == "__main__":
    board = Board.parse_instance(sys.argv[1])
    print("Initial:\n", board.to_string(), sep="")

    problem = Numbrix(board)
    s0 = NumbrixState(board)
    print(problem.actions(s0))

    # s1 = problem.result(s0, (2, 2, 1))
    # s2 = problem.result(s1, (0, 2, 3))
    # s3 = problem.result(s2, (0, 1, 4))
    # s4 = problem.result(s3, (1, 1, 5))
    # s5 = problem.result(s4, (2, 0, 7))
    # s6 = problem.result(s5, (1, 0, 8))
    # goal_node = problem.result(s6, (0, 0, 9))

    # goal_node = breadth_first_tree_search(problem)
    # goal_node = depth_first_tree_search(problem)
    goal_node = greedy_search(problem, problem.h)
    # goal_node = recursive_best_first_search(problem, problem.h)
    # goal_node = astar_search(problem, problem.h, display=True)

    if goal_node != None:
        print("Is goal?", problem.goal_test(goal_node.state))
        print("Solution:\n", goal_node.state.board.to_string(), sep="")
    else:
        print("Found no solution!")