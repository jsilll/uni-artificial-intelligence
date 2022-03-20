# numbrix.py: Template para implementação do projeto de Inteligência Artificial 2021/2022.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# Grupo 19:
# 96915 Tomás Nunes
# 95597 João Silveira

import sys
from copy import deepcopy
from search import Problem, Node, astar_search, breadth_first_tree_search, depth_first_tree_search, greedy_search, recursive_best_first_search


class NumbrixState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = NumbrixState.state_id
        NumbrixState.state_id += 1

    def __lt__(self, other):
        return self.id < other.id

    def fill_number(self, row: int, col: int, number: int):
        """
        Preenche um quadrado no tabuleiro
        """
        state_copy = deepcopy(self)
        state_copy.board.squares[row][col] = number
        state_copy.board.numbers[number - 1] = (row, col)
        state_copy.board.zeros = state_copy.board.zeros - 1
        return state_copy


class Board:
    def __init__(self, N: int, squares: list):
        """
        Representação interna de um tabuleiro de Numbrix.
        """
        self.N = N
        self.zeros = 0
        self.squares = squares
        self.numbers = [None] * N**2

        for i in range(N):
            for j in range(N):
                if squares[i][j] == 0:
                    self.zeros = self.zeros + 1
                else:
                    self.numbers[self.squares[i][j] - 1] = (i, j)

    def get_number(self, row: int, col: int) -> int:
        """
        Devolve o valor na respetiva posição do tabuleiro.
        """
        return self.squares[row][col]

    def adjacent_vertical_numbers(self, row: int, col: int) -> (int, int):
        """
        Devolve os valores imediatamente abaixo e acima,
        respectivamente.
        """
        down = self.squares[row + 1][col] if row + 1 < self.N else None
        up = self.squares[row - 1][col] if row - 1 >= 0 else None
        return (down, up)

    def adjacent_horizontal_numbers(self, row: int, col: int) -> (int, int):
        """
        Devolve os valores imediatamente à esquerda e à direita,
        respectivamente.
        """
        left = self.squares[row][col - 1] if col - 1 >= 0 else None
        right = self.squares[row][col + 1] if col + 1 < self.N else None
        return (left, right)

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
            board = Board(
                N, [[int(num) for num in f.readline().split("\t")] for _ in range(N)])
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
        for i in range(state.board.N):
            for j in range(state.board.N):
                # Para todas as posicoes ainda nao preenchidas
                if state.board.squares[i][j] == 0:

                    neighbors = list(state.board.adjacent_vertical_numbers(i, j)) + list(state.board.adjacent_horizontal_numbers(i, j))
                    valid_neighbors = list(filter(lambda x: x != None and x != 0, neighbors))

                    possible_values = {}
                    for neigh in valid_neighbors:
                        # Este numero nao pode ser N^2
                        # Este numero nao pode ja ter sido usado noutro lado
                        if neigh < state.board.N**2 and state.board.numbers[neigh] == None:
                            if neigh + 1 not in possible_values.keys():
                                possible_values[neigh + 1] = 1
                            else:
                                possible_values[neigh + 1] = possible_values[neigh + 1] + 1
                        # Este numero nao pode ser 1
                        # Este numero nao pode ja ter sido usado noutro lado
                        if neigh != 1 and state.board.numbers[neigh - 2] == None:
                            if neigh - 1 not in possible_values.keys():
                                possible_values[neigh - 1] = 1
                            else:
                                possible_values[neigh - 1] = possible_values[neigh - 1] + 1

                    # Vamos satisfazer pelo menos um vizinho em cada acao
                    # Drawback, o nosso programa nao faz numbrix nao preenchidos
                    
                    # 0 Vizinhos -> Skip, nao estamos a perder solucoes no espaco de todas as solucoes
                    n_neighbors = len(valid_neighbors)
                    if n_neighbors == 0:
                        continue

                    # 1 ou N^2
                    # Satisfazer a exatamente 1
                    if 1 in possible_values.keys():
                        res.append((i, j, 1))
                        del possible_values[1]
                    if state.board.N**2 in possible_values.keys():
                        res.append((i, j, state.board.N**2))
                        del possible_values[state.board.N**2]

                    n_edges = neighbors.count(None)
                    # Numero no meio do tabuleiro (0 Nones)
                    if n_edges == 0:
                        # 1 Vizinho, 2 Vizinhos 3 Vizinhos
                        if n_neighbors <= 3:
                            # Tem de satisfazer pelo menos 1 deles
                            for possible_value in filter(lambda x: x[1] >= 1, possible_values.items()):
                                res.append((i, j, possible_value[0]))
                        # 4 Vizinhos
                        else: 
                            # Satisfazer a exatamente 2
                            for possible_value in filter(lambda x: x[1] == 2, possible_values.items()):
                                res.append((i, j, possible_value[0]))
                    
                    # Numero na edge (1 None)
                    elif n_edges == 1:
                        # 1 vizinhos 2 vizinhos
                        if n_neighbors <= 2:
                            # Tem de satisfazer pelo menos 1 deles
                            for possible_value in filter(lambda x: x[1] >= 1, possible_values.items()):
                                res.append((i, j, possible_value[0]))
                        # 3 vizinhos
                        else:
                            # Satisfazer a exatamente 2
                            for possible_value in filter(lambda x: x[1] == 2, possible_values.items()):
                                res.append((i, j, possible_value[0]))

                    # Numero na esquina (2 Nones)
                    elif n_edges == 2:
                        # Satisfazer sempre o numero de vizinhos que tens
                        for possible_value in filter(lambda x: x[1] == n_neighbors, possible_values.items()):
                            res.append((i, j, possible_value[0]))
                        
        return res

    def result(self, state: NumbrixState, action) -> NumbrixState:
        """ 
        Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state). 
        """
        row, col, number = action
        # Mooshak TODO: bug isto nao acontece sempre com actions validas
        # if action in self.actions(state):
        return state.fill_number(row, col, number)

    def goal_test(self, state: NumbrixState) -> bool:
        """ 
        Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas com uma sequência de números adjacentes. 
        """
        # TODO: mesmo fazer esta funcao por causa do Mooshak
        return state.board.zeros == 0

    def h(self, node: Node):
        """ Função heuristica utilizada para a procura A*. """
        filled_positions = list(filter(lambda x: x != None, node.state.board.numbers))
        
        estimate = 0
        for i in range(len(filled_positions) - 1):
            estimate += abs(filled_positions[i][0] - filled_positions[i + 1][0]) +\
            abs(filled_positions[i][1] - filled_positions[i + 1][1]) - 1
        
        return max(estimate, node.state.board.zeros)


if __name__ == "__main__":
    # Ler o ficheiro de input de sys.argv[1]
    board = Board.parse_instance(sys.argv[1])
    print("Initial:\n", board.to_string(), sep="")

    problem = Numbrix(board)
    s0 = NumbrixState(board)
    print(s0.board.numbers)

    # s1 = problem.result(s0, (2, 2, 1))
    # s2 = problem.result(s1, (0, 2, 3))
    # s3 = problem.result(s2, (0, 1, 4))
    # s4 = problem.result(s3, (1, 1, 5))
    # s5 = problem.result(s4, (2, 0, 7))
    # s6 = problem.result(s5, (1, 0, 8))
    # goal_node = problem.result(s6, (0, 0, 9))

    # goal_node = breadth_first_tree_search(problem)
    # goal_node = depth_first_tree_search(problem)
    # goal_node = greedy_search(problem, problem.h)
    # goal_node = recursive_best_first_search(problem, problem.h)
    goal_node = astar_search(problem, problem.h, display=True)
    print("Is goal?", problem.goal_test(goal_node.state))
    print("Solution:\n", goal_node.state.board.to_string(), sep="")
