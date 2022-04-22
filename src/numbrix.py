# Grupo 19:
# 96915 Tomás Nunes
# 95597 João Silveira

# from matplotlib.pyplot import fill
from itertools import combinations
from search import Problem, Node, astar_search, depth_first_graph_search, greedy_search, breadth_first_tree_search, depth_first_tree_search, greedy_search, recursive_best_first_search
import cProfile
import pstats
import pickle
import sys
class NumbrixState:
    state_id = 0

    def __init__(self, board):
        self.id = NumbrixState.state_id
        NumbrixState.state_id += 1
        self.board = board

    def __lt__(self, other):
        return self.id < other.id


class Board:
    def __init__(self, n: int, squares: list):
        """
        Representação interna de um tabuleiro de Numbrix.
        """
        self.n = n                                                                                  # Constant value
        self.n_squares = n * n                                                                      # Constant value
        self.n_zeros = sum(map(lambda x: x == 0, [square for row in squares for square in row]))    # Number of Zeros on board
        
        self.squares = squares                                                                      # Square Indexed Matrix
        self.numbers = [None] * self.n_squares                                                      # Number Indexed Array
        self.placed_numbers = []                                                                    # Array of already placed number
        self.missing_numbers = []                                                                   # Array of missing numbers
        
        self.update_numbers_array()
        self.update_missing_and_placed_numbers()

    def update_numbers_array(self):
        self.numbers = [None] * self.n_squares
        for row in range(self.n):
            for col in range(self.n):
                if self.squares[row][col] != 0:
                    self.numbers[self.squares[row][col] - 1] = (row, col)

    def update_missing_and_placed_numbers(self):
        self.placed_numbers = []
        self.missing_numbers = []  
        for number in range(self.n_squares):
            if self.numbers[number] == None:
                self.missing_numbers.append(number + 1)
            else:
                self.placed_numbers.append(number + 1)

    def adjacent_vertical_numbers(self, row: int, col: int) -> tuple:
        """
        Devolve os valores imediatamente abaixo e acima,
        respectivamente.
        """
        down = self.squares[row + 1][col] if row + 1 < self.n else None
        up = self.squares[row - 1][col] if row - 1 >= 0 else None
        return (down, up)

    def adjacent_horizontal_numbers(self, row: int, col: int) -> tuple:
        """
        Devolve os valores imediatamente à esquerda e à direita,
        respectivamente.
        """
        left = self.squares[row][col - 1] if col - 1 >= 0 else None
        right = self.squares[row][col + 1] if col + 1 < self.n else None
        return (left, right)

    def adjacent_all_numbers(self, row: int, col: int) -> tuple:
        """
        Devolve os valores imediatamente à esquerda, direita,
        abaixo e acima respetivamente.
        """
        left, right = self.adjacent_horizontal_numbers(row, col)
        down, up = self.adjacent_vertical_numbers(row, col)
        return (left, right, down, up)

    def actions_for_number(self, number : int) -> list:
        """
        Devolve um lista com todas as ações possíveis para um número
        """
        def actions_from_neighbor(number : int, offset: int):
            x1 = self.numbers[number - 1 + offset][0]
            y1 = self.numbers[number - 1 + offset][1]
            left, right, down, up = self.adjacent_all_numbers(x1, y1)
            if left == 0:
                yield (x1, y1 - 1, number)
            if right == 0:
                yield (x1, y1 + 1, number)
            if down == 0:
                yield (x1 + 1, y1, number)
            if up == 0:
                yield (x1 - 1, y1, number)

        possible_left = []
        left_placed = False
        if number != 1 and self.numbers[number - 2] != None:
            left_placed = True
            possible_left = [action for action in actions_from_neighbor(number, -1)]

        possible_right = []
        right_placed = False
        if number != self.n_squares and self.numbers[number] != None:
            right_placed = True
            possible_right = [action for action in actions_from_neighbor(number, 1)]

        if not left_placed and not right_placed:
            return []
        
        if left_placed and right_placed:
            intersection = [num for num in possible_left if num in possible_right]
            res = [action for action in intersection if (self.is_action_distance_possible(action) and self.validate_action(action))]
            return res if res else [None]

        res = [action for action in possible_left + possible_right if (self.is_action_distance_possible(action) and self.validate_action(action))]
        return res if res else [None]

    def validate_action(self, action: tuple) -> bool:
        """
        Checks whether an action is valid or not
        """

        def is_valid_position(position: tuple):
            """
            Checks whether a position is valid or not
            """
            row, col, number = position
            
            if number == None:
                return True

            adj = self.adjacent_all_numbers(row, col)
            
            if number == 0:
                n_adj_nones = len([neigh for neigh in adj if neigh == None])
                adj_numbers = [neigh for neigh in adj if neigh != None and neigh != 0]
                blockers = n_adj_nones + len(adj_numbers)
                if blockers < 3:
                    return True
                elif self.numbers[0] == None and (2 in adj_numbers):
                    return True
                elif self.numbers[self.n_squares - 1] == None and (self.n_squares - 1 in adj_numbers):
                    return True
                elif blockers == 3:
                    if self.numbers[0] == None:
                        # TODO: manhattan distance to right number
                        return True 
                    if self.numbers[self.n_squares - 1] == None:
                        # TODO: manhattan distance to left number
                        return True
                    for adj_num in adj_numbers:
                        if adj_num != 1 and self.numbers[adj_num - 2] == None:
                            # TODO: manhattan distance to adj_num - 2
                            return True                            
                        if adj_num != self.n_squares and self.numbers[adj_num] == None:
                            # TODO: manhattan distance to adj_num + 2
                            return True
                    return False
                for pair1, pair2 in combinations(adj_numbers, 2):
                    if (abs(pair1 - pair2) == 2) and ((min(pair1, pair2) + 1) in self.missing_numbers):
                        return True
                return False

            n_adj_zeros = adj.count(0)
            n_adj_valid = len([neigh for neigh in adj if neigh != None and neigh != 0 and abs(number - neigh) == 1])
            
            if n_adj_valid == 2 or n_adj_zeros >= 2:
                return True
            elif n_adj_valid == 1 and n_adj_zeros > 0:
                return True
            elif (number == 1 or number == self.n_squares) and (n_adj_valid == 1 or n_adj_zeros == 1):
                return True
            return False

        row, col, number = action
        if is_valid_position(action):
            self.squares[row][col] = number
            left, right, down, up = self.adjacent_all_numbers(row, col)
            if (is_valid_position((row, col - 1, left))  \
            and is_valid_position((row, col + 1, right)) \
            and is_valid_position((row + 1, col, down))  \
            and is_valid_position((row - 1, col, up))):
                self.squares[row][col] = 0
                return True
        
        self.squares[row][col] = 0
        return False

    def is_action_distance_possible(self, action : tuple):
        """
        Check if manhattan distance of new action 
        allows to complete the board successfuly
        """

        def find_numerically_closest(number : int):
            closest_left = None
            i = number - 1
            while i >= 0:
                if self.numbers[i] != None:
                    closest_left = self.numbers[i]
                    break
                i = i - 1 
            
            closest_right = None
            i = number - 1
            while i < self.n_squares:
                if self.numbers[i] != None:
                    closest_right = self.numbers[i]
                    break
                i = i + 1 
            
            return (closest_left, closest_right)

        def is_manhattan_possible(action : tuple, closest : tuple):
            row, col, number = action
            row_closest, col_closest, number_closest = closest
            manhattan_distance = abs(row - row_closest) + abs(col - col_closest)
            numeric_distance = abs(number - number_closest)
            if manhattan_distance > numeric_distance:
                return False
            return True

        _, _, number = action
        closest_left, closest_right = find_numerically_closest(number)
 
        if closest_left: 
            row_left, col_left = closest_left
            if not is_manhattan_possible(action, (row_left, col_left, self.squares[row_left][col_left])):
                return False

        if closest_right:
            row_right, col_right = closest_right
            if not is_manhattan_possible(action, (row_right, col_right, self.squares[row_right][col_right])):
                return False

        return True

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
        if state.board.n_zeros == 0:
            return []

        minimum_actions = []
        n_minimum_actions = float("inf")
        for number in state.board.missing_numbers:
            actions = state.board.actions_for_number(number)
            n_actions = len(actions)
            if actions == [None]:
                return [] 
            elif n_actions == 1:
                return actions
            elif n_actions != 0 and n_actions < n_minimum_actions:
                n_minimum_actions = n_actions
                minimum_actions = actions
        return minimum_actions

    def result(self, state: NumbrixState, action) -> NumbrixState:
        """
        Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state).
        """
        row, col, number = action

        state_copy = pickle.loads(pickle.dumps(state, - 1))
        state_copy.board.n_zeros = state_copy.board.n_zeros - 1       
        state_copy.board.squares[row][col] = number       
        state_copy.board.numbers[number - 1] = (row, col)
        state_copy.board.missing_numbers.remove(number)
        state_copy.board.placed_numbers.append(number)

        return state_copy

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
            manhattan_distance = abs(x1 - x2) + abs(y1 - y2) - 1
            if manhattan_distance > 1:
                return False
        return True

    def h(self, node: Node):
        """
        Função heuristica utilizada para a procura A*.
        """
        zeros_sum = 0
        for row in range(node.state.board.n):
            for col in range(node.state.board.n):
                if node.state.board.squares[row][col] == 0:
                    zeros_sum += len([adj for adj in node.state.board.adjacent_all_numbers(row, col) if adj == 0])

        return zeros_sum

if __name__ == "__main__":
    board : Board = Board.parse_instance(sys.argv[1])
    problem : Problem = Numbrix(board)
    with cProfile.Profile() as profile:
        goal_node = greedy_search(problem)
        stats = pstats.Stats(profile)
        stats.sort_stats(pstats.SortKey.TIME)
        stats.print_stats()
    # goal_node = greedy_search(problem)
    if goal_node != None:
        print(goal_node.state.board.to_string(), sep="")
    else:
        print("Found no solution!")