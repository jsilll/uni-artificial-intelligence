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

def consecutive_nones_indexes(x : list) -> list:
    out = []
    count = 0
    for i in range(len(x)):
        if x[i] == None:
            count += 1
        elif count:
            out.append([i - count, i - 1])
            count = 0
    if count:
        out.append([i - count + 1, i])
    return out

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

        copy = deepcopy(self)
        copy.board.squares[row][col] = number
        copy.board.numbers[number - 1] = (row, col)
        copy.board.zeros = copy.board.zeros - 1
        copy.board.unused_segments = consecutive_nones_indexes(copy.board.numbers)

        return copy

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

        # TODO: Melhorar
        for i in range(self.n):
            for j in range(self.n):
                if squares[i][j] != 0:
                    self.numbers[self.squares[i][j] - 1] = (i, j)

        self.unused_segments = consecutive_nones_indexes(self.numbers)


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
        """
        Devolve os valores imediatamente à esquerda, direita,
        cima e baixo respetivamente.
        """

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

        def actions_for_value(chosen_number):
            if chosen_number != 1 and state.board.numbers[chosen_number - 2] != None:
                x1 = state.board.numbers[chosen_number - 2][0]
                y1 = state.board.numbers[chosen_number - 2][1]
                
                left, right, down, up = state.board.get_neighbors(x1, y1)
                if left == 0:
                    yield (x1, y1 - 1, chosen_number)  
                if right == 0:
                    yield (x1, y1 + 1, chosen_number)  
                if down == 0:
                    yield (x1 + 1, y1, chosen_number)  
                if up == 0:
                    yield (x1 - 1, y1, chosen_number)

            if chosen_number != state.board.n_squares and state.board.numbers[chosen_number] != None:
                x1 = state.board.numbers[chosen_number][0]
                y1 = state.board.numbers[chosen_number][1]

                left, right, down, up = state.board.get_neighbors(x1, y1)
                if left == 0:
                    yield (x1, y1 - 1, chosen_number)  
                if right == 0:
                    yield (x1, y1 + 1, chosen_number)  
                if down == 0:
                    yield (x1 + 1, y1, chosen_number)  
                if up == 0:
                    yield (x1 - 1, y1, chosen_number)

        def select_next_number():
            min_segment_size = float("inf")
            for i in range(len(state.board.unused_segments)):
                seg_size = state.board.unused_segments[i][1] - state.board.unused_segments[i][0]
                if seg_size < min_segment_size:
                    chosen_numbers = state.board.unused_segments[i]
                    min_segment_size = seg_size
            
            left_candidate = chosen_numbers[0] + 1
            right_candidate = chosen_numbers[1] + 1

            if min_segment_size == 0:
                return left_candidate
            elif left_candidate == 1:
                return right_candidate
            elif right_candidate == state.board.n_squares:
                return left_candidate
            else:
                return left_candidate

        selected_number = select_next_number()
        return [action for action in actions_for_value(selected_number)]

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
            
        for i in range(state.board.n_squares - 1): 
            x1 = state.board.numbers[i][0]
            y1 = state.board.numbers[i][1]
            x2 = state.board.numbers[i + 1][0]
            y2 = state.board.numbers[i + 1][1]
            
            man_distance = abs(x1 - x2) + abs(y1 - y2) - 1
            if man_distance > 1:
                return False

        return True

    def h(self, node: Node):
        """
        Função heuristica utilizada para a procura A*.
        """

        # TODO: performance can be improved, iterate only on number gaps
        filled_positions = [x for x in  node.state.board.numbers if x != None]
        for i in range(len(filled_positions) - 1):
            x1 = filled_positions[i][0]
            y1 = filled_positions[i][1]
            x2 = filled_positions[i + 1][0]
            y2 = filled_positions[i + 1][1]

            num_distance = abs(node.state.board.squares[x1][y1] - node.state.board.squares[x2][y2])
            man_distance = abs(x1 - x2) + abs(y1 - y2)
            if man_distance > num_distance:
                return float("inf")

        # Like an early goal test ??
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

        # Completely surrounded zeros that are impossible to fill
        for i in range(node.state.board.n):
            for j in range(node.state.board.n):
                if node.state.board.squares[i][j] == 0:
                    neighbors = node.state.board.get_neighbors(i, j)
                    if len(list(filter(lambda x : x != 0, neighbors))) == 4:
                        num_neighbors = list(filter(lambda x : x != 0 and x != None, neighbors))
                        n = len(num_neighbors)
                        diff = float("inf")
                        for k in range(n - 1):
                            for l in range(k + 1, n):
                                diff = min(abs(num_neighbors[k] - num_neighbors[l]), diff)
                        if diff != 2:
                            return float("inf")

        return node.state.board.zeros


if __name__ == "__main__":
    board = Board.parse_instance(sys.argv[1])
    # print("Initial:\n", board.to_string(), sep="")

    problem = Numbrix(board)
    s0 = NumbrixState(board)
    # print(problem.actions(s0))

    # print(s0.board.numbers)
    # print(s0.board.unused_segments)

    # s1 = problem.result(s0, (2, 2, 1))
    # s2 = problem.result(s1, (0, 2, 3))
    # s3 = problem.result(s2, (0, 1, 4))
    # s4 = problem.result(s3, (1, 1, 5))
    # s5 = problem.result(s4, (2, 0, 7))
    # s6 = problem.result(s5, (1, 0, 8))
    # goal_node = problem.result(s6, (0, 0, 9))

    # goal_node = breadth_first_tree_search(problem)
    # goal_node = depth_first_tree_search(problem)
    # goal_node = recursive_best_first_search(problem, problem.h)

    # goal_node = astar_search(problem, problem.h, display=True)
    goal_node = greedy_search(problem, problem.h)

    if goal_node != None:
        # print("Is goal?", problem.goal_test(goal_node.state))
        # print("Solution:\n", goal_node.state.board.to_string(), sep="")
        print(goal_node.state.board.to_string())
    else:
        print("Found no solution!")