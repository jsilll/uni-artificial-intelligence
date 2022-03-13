# numbrix.py: Template para implementação do projeto de Inteligência Artificial 2021/2022.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# Grupo 19:
# 96915 Tomás Nunes
# 95597 João Silveira

from copy import deepcopy
import sys
from search import Problem, Node, astar_search, breadth_first_tree_search, depth_first_tree_search, greedy_search, recursive_best_first_search


class NumbrixState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = NumbrixState.state_id
        NumbrixState.state_id += 1

    def __lt__(self, other):
        return self.id < other.id

    # TODO: outros metodos da classe


class Board:
    """
    Representação interna de um tabuleiro de Numbrix.
    """

    def __init__(self, N: int, numbers: dict):
        self.N = N
        self.numbers = numbers

    def get_number(self, row: int, col: int) -> int:
        """
        Devolve o valor na respetiva posição do tabuleiro.
        """
        return self.numbers[(row, col)] if (row, col) in self.numbers.keys() else 0

    def adjacent_vertical_numbers(self, row: int, col: int) -> (int, int):
        """
        Devolve os valores imediatamente abaixo e acima,
        respectivamente.
        """
        return (self.numbers[(row + 1, col)] if row + 1 < self.N else None, self.numbers[(row - 1, col)] if row - 1 >= 0 else None)

    def adjacent_horizontal_numbers(self, row: int, col: int) -> (int, int):
        """
        Devolve os valores imediatamente à esquerda e à direita,
        respectivamente.
        """
        return (self.numbers[(row, col - 1)] if col - 1 >= 0 else None, self.numbers[(row, col + 1)] if col + 1 < self.N else None)

    def to_string(self):
        return '\n'.join([''.join(['{:4}'.format(self.numbers[(i, j)]) for j in range(self.N)]) for i in range(self.N)])

    @ staticmethod
    def parse_instance(filename: str):
        """
        Lê o ficheiro cujo caminho é passado como argumento e retorna
        uma instância da classe Board.
        """
        with open(filename, "r") as f:
            N = int(f.readline())
            numbers = {}
            for i in range(N):
                line = f.readline().split("\t")
                for j in range(N):
                    if line[j] != 0:
                        numbers[(i, j)] = int(line[j])
            board = Board(N, numbers)
            f.close()

        return board

    # TODO: outros metodos da classe


class Numbrix(Problem):
    def __init__(self, board: Board):
        """ O construtor especifica o estado inicial. """
        # TODO
        pass

    def actions(self, state: NumbrixState):
        """ 
        Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento. 
        """
        # for i linhas
        #   for j in colunas
        #       if board[i][j] == 0:
        #           for adj_v get posible1
        #           for adj_h get posible 2
        #           return intersect posible1 posible2
        # TODO
        pass

    def result(self, state: NumbrixState, action):
        """ 
        Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state). 
        """
        # TODO
        pass

    def goal_test(self, state: NumbrixState):
        """ Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas com uma sequência de números adjacentes. """
        # TODO
        pass

    def h(self, node: Node):
        """ Função heuristica utilizada para a procura A*. """
        # TODO
        pass

    # TODO: outros metodos da classe


if __name__ == "__main__":
    # Ler o ficheiro de input de sys.argv[1]
    board = Board.parse_instance(sys.argv[1])
    print("Initial:\n", board.to_string(), sep="")

    print(f"Vertical Adj. {board.adjacent_vertical_numbers(1,1)}")
    print(f"Horizontal Adj. {board.adjacent_horizontal_numbers(1,1)}")

    problem = Numbrix(board)
    s0 = NumbrixState(board)
    s0.board.get_number(2, 2)
    result_state = problem.result(s0, (2, 2, 1))

    # TODO:
    # Usar uma técnica de procura para resolver a instância,
    # Retirar a solução a partir do nó resultante,
    # Imprimir para o standard output no formato indicado.

    # goal_node = astar_search(problem)
    s1 = problem.result(s0, (2, 2, 1))
    s2 = problem.result(s1, (0, 2, 3))
    s3 = problem.result(s2, (0, 1, 4))
    s4 = problem.result(s3, (1, 1, 5))
    s5 = problem.result(s4, (2, 0, 7))
    s6 = problem.result(s5, (1, 0, 8))

    goal_node = problem.result(s6, (0, 0, 9))
    print("Is goal?", problem.goal_test(goal_node))
    print("Solution:\n", board.to_string(), sep="")
