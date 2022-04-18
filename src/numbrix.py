# Grupo 19:
# 96915 Tomás Nunes
# 95597 João Silveira

from ssl import ALERT_DESCRIPTION_BAD_CERTIFICATE_STATUS_RESPONSE
import _pickle as pickle
import cProfile
import pstats
import sys
from copy import deepcopy
from search import Problem, Node, astar_search, depth_first_graph_search, greedy_search, breadth_first_tree_search, depth_first_tree_search, greedy_search, recursive_best_first_search

# from matplotlib.pyplot import fill

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

        self.placed_numbers = []
        self.missing_numbers = []
        for number in range(self.n_squares):
            if self.numbers[number] == None:
                self.missing_numbers.append(number + 1)
            else:
                self.placed_numbers.append(number + 1)

    def adjacent_vertical_numbers(self, row: int, col: int) -> tuple[int, int]:
        """
        Devolve os valores imediatamente abaixo e acima,
        respectivamente.
        """
        return (self.squares[row + 1][col] if row + 1 < self.n else None, self.squares[row - 1][col] if row - 1 >= 0 else None)

    def adjacent_horizontal_numbers(self, row: int, col: int) -> tuple[int, int]:
        """
        Devolve os valores imediatamente à esquerda e à direita,
        respectivamente.
        """
        return (self.squares[row][col - 1] if col - 1 >= 0 else None, self.squares[row][col + 1] if col + 1 < self.n else None)

    def adjacent_all_numbers(self, row: int, col: int) -> tuple[int, int]:
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

    def actions_for_number(self, number) -> list:
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

        intersection = [number for number in possible_left if number in possible_right] # TODO: improve complexity
        if intersection == [] and left_placed and right_placed:
            return [None]
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
        minimum_actions = []
        n_minimum_actions = float("inf")
        for number in state.board.missing_numbers:
            actions = state.board.actions_for_number(number)
            n_actions = len(actions)
            if actions == [None]:
                return []
            if n_actions == 1:
                res = [action for action in actions if (self.action_distance_possible(state, action))]
                return res # and self.testPositions(state.board, action))] # and self.testPositions(state.board, action))]
            if n_actions != 0 and n_actions < n_minimum_actions:
                n_minimum_actions = n_actions
                minimum_actions = actions
        res = [action for action in minimum_actions if (self.action_distance_possible(state, action))]
        return res # and self.testPositions(state.board, action))] # and self.testPositions(state.board, action))]

    def testPositions(self, board:Board, action:tuple) -> bool:

        def test_helper(action:tuple , board:Board):
            row, col, number = action
            lists = board.adjacent_all_numbers(row, col)
            n_zeros = lists.count(0)
            valid_neighbors = len([number for number in lists if number != None and (number == number + 1 or number == number - 1) and number !=0])
            if valid_neighbors == 2 or n_zeros >= 2:
                return True
            elif valid_neighbors == 1 and n_zeros != 0:
                return True
            elif (number == 1 or number == board.n_squares) and valid_neighbors == 1:
                return True
            elif (number == 1 or number == board.n_squares) and n_zeros == 1:
                return True
            return False

        row, col, number = action
        a = 0
        
        # Check if n matches the conditions to fill the position (x,y)
        if test_helper(action, board):
            a=1

        # Force n to be in that position (x,y)
        board.squares[row][col] = number

        # Adjacent numbers (left and up) of n 
        left = board.adjacent_horizontal_numbers(row,col)[0] 
        up =  board.adjacent_vertical_numbers(row,col)[1]

        # Get the tuples with the adjacent numbers
        check_left = (row,col - 1,left)
        check_up = (row - 1,col,up)

        # If p matches the conditions to fill the position (a == 1) and once we forced n to be in that position (x,y), then
        # We try to check the adjacent numbers of the adjacents - left and up - of (x,y,n) = p
        if(a and test_helper(check_left, board) and test_helper(check_up,board)):
            board.squares[row][col] = 0
            return True
            
        # If the adjacent numbers of the adjacents of (x,y,n) = p don't match the conditions 
        board.squares[row][col] = 0
        return False

    def action_distance_possible(self, state: NumbrixState, action : tuple):
        row1, col1, number1 = action
        for number2 in state.board.placed_numbers:
            row2, col2 = state.board.numbers[number2 - 1]
            manhattan_distance = abs(row1 - row2) + abs(col1 - col2)
            numeric_distance = abs(number1 - number2)
            if manhattan_distance > numeric_distance:
                return False  
        return True

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
        goal_node = depth_first_graph_search(problem)
        stats = pstats.Stats(profile)
        stats.sort_stats(pstats.SortKey.TIME)
        stats.print_stats()
    if goal_node != None:
        print(goal_node.state.board.to_string(), sep="")
    else:
        print("Found no solution!")