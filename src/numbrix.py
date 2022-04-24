# Grupo 19:
# 96915 Tomás Nunes
# 95597 João Silveira

import math
from pickle import loads, dumps
from itertools import combinations
import sys
from search import Problem, Node, astar_search, breadth_first_tree_search, depth_first_graph_search,\
    depth_first_tree_search, greedy_search, recursive_best_first_search

class NumbrixState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = NumbrixState.state_id
        NumbrixState.state_id += 1

    def __lt__(self, other):
        return self.id < other.id

class Board:
    def __init__(self, N: int, squares: list):
        """ 
        Representação interna de um tabuleiro de Numbrix. 
        """
        self.N = N
        self.N_SQUARES = N**2

        self.n_zeros = sum(map(lambda x: x == 0, [square for row in squares for square in row]))

        self.squares = squares
        self.numbers = [None] * self.N_SQUARES
        self.placed_numbers = list()
        self.missing_numbers = list()

        # Filling Numbers List
        for row in range(self.N):
            for col in range(self.N):
                if self.squares[row][col] != 0:
                    self.numbers[self.squares[row][col] - 1] = (row, col)

        # Filling Placed and Missing Numbers Lists
        for num in range(1, self.N_SQUARES + 1):
            if self.numbers[num - 1] == None:
                self.missing_numbers.append(num)
            else:
                self.placed_numbers.append(num)

    def get_number(self, row: int, col: int) -> int:
        """ 
        Devolve o valor na respetiva posição do tabuleiro. 
        """
        return self.squares[row][col]

    def get_position(self, num : int):
        """ 
        Devolve a posicao de um numero no tabuleiro 
        """
        return self.numbers[num - 1]
    
    def adjacent_vertical_numbers(self, row: int, col: int) -> tuple:
        """ 
        Devolve os valores imediatamente abaixo e acima, 
        respectivamente. 
        """
        down = self.squares[row + 1][col] if row + 1 < self.N else None
        up = self.squares[row - 1][col] if row - 1 >= 0 else None
        return (down, up)
    
    def adjacent_horizontal_numbers(self, row: int, col: int) -> tuple:
        """ 
        Devolve os valores imediatamente à esquerda e à direita, 
        respectivamente. 
        """
        left = self.squares[row][col - 1] if col - 1 >= 0 else None
        right = self.squares[row][col + 1] if col + 1 < self.N else None
        return (left, right)
    
    def adjacent_all_numbers(self, row: int, col: int) -> tuple:
        """ 
        Devolve os valores imediatamente à esquerda, direita, 
        abaixo e acima respetivamente.
        """
        left, right = self.adjacent_horizontal_numbers(row, col)
        down, up = self.adjacent_vertical_numbers(row, col)
        return (left, right, down, up)

    def get_numerically_closest(self, number : int) -> tuple:
        """
        Devolve um tuplo que contem os numeros numericamente mais proximos
        "acima" e a "abaixo" de um dado numero.
        """
        closest_left = None
        if number != 1:
            for num in range(number - 1, 0, -1):
                if self.get_position(num) != None:
                    closest_left = num
                    break
        
        closest_right = None
        if number != self.N_SQUARES:
            for num in range(number + 1, self.N_SQUARES + 1):
                if self.get_position(num) != None:
                    closest_right = num
                    break

        return (closest_left, closest_right)

    def actions_for_number(self, number : int) -> list:
        """
        Devolve um lista com todas as ações possíveis para um número
        """

        def actions_given_by_numeric_neighbor(number : int, right : bool):
            """
            Gera todas as posicoes livres para um vizinho numerico de um numero
            """
            offset = 1 if right else -1

            if self.get_position(number + offset) != None:
                row, col = self.get_position(number + offset)
                left, right, down, up = self.adjacent_all_numbers(row, col)
                if left == 0:
                    yield (row, col - 1, number)
                if right == 0:
                    yield (row, col + 1, number)
                if down == 0:
                    yield (row + 1, col, number)
                if up == 0:
                    yield (row - 1, col, number)

        is_left_placed = False
        possible_left = list()
        if number != 1 and self.get_position(number - 1) != None:
            is_left_placed = True
            possible_left = [action for action in actions_given_by_numeric_neighbor(number, False)]

        is_right_placed = False
        possible_right = list()
        if number != self.N_SQUARES and self.get_position(number + 1) != None:
            is_right_placed = True
            possible_right = [action for action in actions_given_by_numeric_neighbor(number, True)]

        # Filtering Invalid Actions
        if not is_left_placed and not is_right_placed:
            return list()

        joint_actions = list()
        if is_left_placed and is_right_placed:
            joint_actions = [action for action in possible_left if action in possible_right]
        else:
            joint_actions = possible_left + possible_right
        
        joint_actions_filtered = [action for action in joint_actions if\
            (self.is_action_distances_possible(*action) and self.is_valid_action(*action))]
        return joint_actions_filtered if joint_actions_filtered else [None]

    def is_valid_action(self, row : int, col : int, number : int) -> bool:
        """
        Verifica se uma ação é válida ou nao
        """
        if self.is_position_valid(row, col):
            left, right, down, up = self.adjacent_all_numbers(row, col)
            self.squares[row][col] = number
            self.numbers[number - 1] = (row, col)
            if left != None and not self.is_position_valid(row, col - 1):
                self.squares[row][col] = 0
                self.numbers[number - 1] = None
                return False
            if right != None and not self.is_position_valid(row, col + 1):
                self.squares[row][col] = 0
                self.numbers[number - 1] = None
                return False
            if down != None and not self.is_position_valid(row + 1, col):
                self.squares[row][col] = 0
                self.numbers[number - 1] = None
                return False
            if up != None and not self.is_position_valid(row - 1, col):
                self.squares[row][col] = 0
                self.numbers[number - 1] = None
                return False
            self.squares[row][col] = 0
            self.numbers[number - 1] = None
            return True
        self.squares[row][col] = 0
        self.numbers[number - 1] = None
        return False

    def is_position_valid(self, row : int, col : int) -> bool:
        """
        Verifica se uma posicao e valida
        """
        number = self.get_number(row, col)
        neighbors = self.adjacent_all_numbers(row, col)

        if number == 0:
            neigh_nums = [neigh for neigh in neighbors if neigh != None and neigh != 0]
            blocked_squares = 4 - neighbors.count(0)

            if blocked_squares < 3:
                return True
            elif self.get_position(1) == None and (min(2, self.N_SQUARES) in neigh_nums):
                return True
            elif self.get_position(self.N_SQUARES) == None and (max(1, self.N_SQUARES - 1) in neigh_nums):
                return True
            elif blocked_squares == 3:
                if self.get_position(1) == None and self.is_action_distances_possible(row, col, 1):
                    return True
                if self.get_position(self.N_SQUARES) == None and self.is_action_distances_possible(row, col, self.N_SQUARES):
                    return True
                for neigh in neigh_nums:
                    if neigh != 1 and self.get_position(neigh - 1) == None and self.is_action_distances_possible(row, col, neigh - 1):
                        return True
                    if neigh != self.N_SQUARES and self.get_position(neigh + 1) == None and self.is_action_distances_possible(row, col, neigh + 1):
                        return True
                return False
            for neigh1, neigh2 in combinations(neigh_nums, 2):
                numeric_diff = abs(neigh1 - neigh2)
                if numeric_diff == 2 and self.get_position(min(neigh1, neigh2) + 1) == None:
                    return True
            return False

        n_neigh_zero = neighbors.count(0)
        n_neigh_valid = len([neigh for neigh in neighbors if neigh != None and neigh != 0 and abs(number - neigh) == 1]) 

        if n_neigh_valid == 2 or n_neigh_zero >= 2:
            return True
        elif n_neigh_valid == 1 and n_neigh_zero > 0:
            return True
        elif (number == 1 or number == self.N_SQUARES) and (n_neigh_valid == 1 or n_neigh_zero == 1):
            return True
        return False

    def is_action_distances_possible(self, row : int, col : int, number : int):
        """
        Verifica se a distancia de manhattan de um numero numa certa posicao
        ate aos seus numeros numericamente vizinhos permite que seja feita
        uma ligacao
        """

        closest_left, closest_right = self.get_numerically_closest(number)

        if closest_left:
            row_left, col_left = self.get_position(closest_left)
            numeric_distance = abs(closest_left - number)
            manhattan_distance = abs(row_left - row) + abs(col_left - col)
            if manhattan_distance > numeric_distance:
                return False
        
        if closest_right:
            row_right, col_right = self.get_position(closest_right)
            numeric_distance = abs(closest_right - number)
            manhattan_distance = abs(row_right - row) + abs(col_right - col)
            if manhattan_distance > numeric_distance:
                return False

        return True

    def to_string(self):
        """
        Representacao ascii do tabuleiro
        """
        return '\n'.join(['\t'.join([str(number) for number in row]) for row in self.squares])

    def __str__(self):
        return self.to_string()

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
        """ O construtor especifica o estado inicial. """
        self.initial = NumbrixState(board)

    def actions(self, state: NumbrixState):
        """ 
        Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento. 
        """

        if state.board.n_zeros == 0:
            return []

        for row in range(state.board.N):
            for col in range(state.board.N):
                if state.board.get_number(row, col) == 0:
                    neighbors = state.board.adjacent_all_numbers(row, col)
                    if neighbors.count(0) < 2:
                        left, right, down, up  = neighbors
                        if left != None and not state.board.is_position_valid(row, col - 1):
                            return []
                        elif right != None and not state.board.is_position_valid(row, col + 1):
                            return []
                        elif down != None and not state.board.is_position_valid(row + 1, col):
                            return []
                        elif up != None and not state.board.is_position_valid(row - 1, col):
                            return []
                    
        least_actions = []
        n_least_actions = math.inf
        for num in state.board.missing_numbers:
            actions = state.board.actions_for_number(num)
            n_actions = len(actions)
            if actions == [None]:
                return [] 
            elif n_actions == 1:
                return actions
            elif n_actions != 0 and n_actions < n_least_actions:
                n_least_actions = n_actions
                least_actions = actions
        return least_actions

    def result(self, state: NumbrixState, action):
        """ 
        Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de 
        self.actions(state). 
        """
        row, col, number = action

        state_copy = loads(dumps(state, - 1))
        state_copy.board.n_zeros = state_copy.board.n_zeros - 1       
        state_copy.board.squares[row][col] = number       
        state_copy.board.numbers[number - 1] = (row, col)
        state_copy.board.missing_numbers.remove(number)
        state_copy.board.placed_numbers.append(number)

        return state_copy

    def goal_test(self, state: NumbrixState):
        """ 
        Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro 
        estão preenchidas com uma sequência de números adjacentes. 
        """
        if state.board.n_zeros != 0:
            return False
        for num in range(1, state.board.N_SQUARES):
            row1, col1 = state.board.get_position(num)
            row2, col2 = state.board.get_position(num + 1)
            manhattan_distance = abs(row1 - row2) + abs(col1 - col2)
            if manhattan_distance != 1:
                return False
        return True

    def h(self, node: Node):
        """ Função heuristica utilizada para a procura A*. """
        zeros_sum = 0
        for row in range(node.state.board.N):
            for col in range(node.state.board.N):
                if node.state.board.get_number(row, col) == 0:
                    zeros_sum += len([adj for adj in node.state.board.adjacent_all_numbers(row, col) if adj == 0])

        return max(0, node.state.board.n_zeros - 0.1 * zeros_sum)

def run_profiled(search_algo):
    import cProfile
    import pstats
    
    board = Board.parse_instance(sys.argv[1])
    problem = Numbrix(board)
    with cProfile.Profile() as profile:
        if "--display" in sys.argv:
            goal_node = search_algo(problem, display=True)
        else:
            goal_node = search_algo(problem, display=False)
        stats = pstats.Stats(profile)
        stats.sort_stats(pstats.SortKey.TIME)
        stats.print_stats()
    if goal_node != None:
        print(goal_node.state.board, sep="")
    else:
        print("Found no solution!")


if __name__ == "__main__":
    algo_names = ["--dfs", "--bfs", "--greedy", "--astar"]
    algos = [depth_first_tree_search, breadth_first_tree_search, greedy_search, astar_search]
    search_algo = depth_first_tree_search

    for name, algo in zip(algo_names, algos):
        if name in sys.argv:
            search_algo = algo
            break

    if "--profiled" in sys.argv:
        run_profiled(search_algo)
    else:
        board = Board.parse_instance(sys.argv[1])
        problem = Numbrix(board)
        if "--display" in sys.argv:
            goal_node = search_algo(problem, display=True)
        else:
            goal_node = search_algo(problem, display=False)
        if goal_node != None:
            print(goal_node.state.board, sep="")
        else:
            print("Found no solution!")
