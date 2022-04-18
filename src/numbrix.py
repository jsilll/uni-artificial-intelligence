# Grupo 19:
# 96915 Tomás Nunes
# 95597 João Silveira

# from copy import deepcopy
# from matplotlib.pyplot import fill
from search import Problem, Node, astar_search, depth_first_graph_search, greedy_search, breadth_first_tree_search, depth_first_tree_search, greedy_search, recursive_best_first_search
import _pickle as pickle
import cProfile
import pstats
import sys

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
        copy = pickle.loads(pickle.dumps(self, - 1))
        copy.board.fill(row, col, number)
        return copy


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
        for row in range(self.n):
            for col in range(self.n):
                if squares[row][col] != 0:
                    self.numbers[self.squares[row][col] - 1] = (row, col)

        self.placed_numbers = []                                                                    # Array of already placed number
        self.missing_numbers = []                                                                   # Array of missing numbers
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
        return (self.squares[row + 1][col] if row + 1 < self.n else None, self.squares[row - 1][col] if row - 1 >= 0 else None)

    def adjacent_horizontal_numbers(self, row: int, col: int) -> tuple:
        """
        Devolve os valores imediatamente à esquerda e à direita,
        respectivamente.
        """
        return (self.squares[row][col - 1] if col - 1 >= 0 else None, self.squares[row][col + 1] if col + 1 < self.n else None)

    def adjacent_all_numbers(self, row: int, col: int) -> tuple:
        """
        Devolve os valores imediatamente à esquerda, direita,
        cima e baixo respetivamente.
        """
        left, right = self.adjacent_horizontal_numbers(row, col)
        down, up = self.adjacent_vertical_numbers(row, col)
        return (left, right, down, up)

    def fill(self, row: int, col: int, number: int) -> None:
        """
        Preenche o tabuleiro com um numero numa dada posicao
        """
        self.n_zeros = self.n_zeros - 1       
        self.squares[row][col] = number       
        self.numbers[number - 1] = (row, col)
        self.missing_numbers.remove(number)
        self.placed_numbers.append(number)

    def actions_for_number(self, number : int) -> list:
        """
        Devolve um lista com todas as ações possíveis para um número
        """
        possible_left = []
        left_placed = False
        if number != 1 and self.numbers[number - 2] != None:
            left_placed = True
            
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
        right_placed = False
        if number != self.n_squares and self.numbers[number] != None:
            right_placed = True
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

        if not left_placed and not right_placed:
            return []

        if left_placed and right_placed:
            intersection = [num for num in possible_left if num in possible_right]
            res = [action for action in intersection if (self.validate_action_distance(action) and self.validate_action(action))]
            return res if res else [None]

        res = [action for action in possible_left + possible_right if (self.validate_action_distance(action) and self.validate_action(action))]
        return res if res else [None]

    def validate_action(self, action:tuple) -> bool:
        """
        Validates an action
        """

        def validate_position(action:tuple):
            """
            Validates a position
            """
            row, col, number = action
            if number:
                adj = self.adjacent_all_numbers(row, col)
                n_zeros = adj.count(0)
                n_valid_neighbors = len([number for neigh in adj if neigh != None and neigh != 0 and abs(number - neigh) == 1])
                if n_valid_neighbors == 2 or n_zeros >= 2:
                    return True
                elif n_valid_neighbors == 1 and n_zeros > 0:
                    return True
                elif (number == 1 or number == self.n_squares) and (n_valid_neighbors == 1 or n_zeros == 1):
                    return True
                
                return False
            
            return True

        row, col, number = action
        if validate_position(action):
            self.squares[row][col] = number
            left, right, down, up = self.adjacent_all_numbers(row, col)
            check_left = (row, col - 1, left)
            check_right = (row, col + 1, right)
            check_down = (row + 1, col, down)
            check_up = (row - 1, col, up)
            
            if (validate_position(check_left) and validate_position(check_right) and validate_position(check_down) and validate_position(check_up)):
                self.squares[row][col] = 0
                return True
        
        self.squares[row][col] = 0
        return False

    def validate_action_distance(self, action : tuple):
        """
        Check if manhattan distance of new action 
        allows to complete the board successfuly
        """
        row1, col1, number1 = action
        for number2 in self.placed_numbers:
            row2, col2 = self.numbers[number2 - 1]
            manhattan_distance = abs(row1 - row2) + abs(col1 - col2)
            numeric_distance = abs(number1 - number2)
            if manhattan_distance > numeric_distance:
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
            if n_actions == 1:
                return actions
            if n_actions != 0 and n_actions < n_minimum_actions:
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
        return state.fill(action[0], action[1], action[2])

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
        return node.state.board.n_zeros

if __name__ == "__main__":
    board : Board = Board.parse_instance(sys.argv[1])
    problem : Problem = Numbrix(board)
    with cProfile.Profile() as profile:
        goal_node = depth_first_tree_search(problem)
        stats = pstats.Stats(profile)
        stats.sort_stats(pstats.SortKey.TIME)
        stats.print_stats()
    if goal_node != None:
        print(goal_node.state.board.to_string(), sep="")
    else:
        print("Found no solution!")