import chess
from scores import *
from chess.engine import Cp, Mate, MateGiven
import time
movetime = None
def is_on_long_diagonal(square):
    return chess.square_file(square) == chess.square_rank(square)

def analyze_chess_position(board):
    trapped_rooks = []
    hanging_pieces = []
    long_diagonal_bishops = []
    unprotected_queens = []
    weak_queens = []

    for square in chess.SQUARES:
        piece = board.piece_at(square)

        if piece is not None:
            if piece.piece_type == chess.ROOK and board.is_pinned(board.turn, square):
                trapped_rooks.append(chess.square_name(square))

            if piece.piece_type == chess.BISHOP and is_on_long_diagonal(square):
                long_diagonal_bishops.append(chess.square_name(square))

            # Check for hanging pieces
            if len(board.attackers(not board.turn, square))>0:
                if board.piece_at(square).color == board.turn:
                    hanging_pieces.append(chess.square_name(square))

            # Check for unprotected queens
            if piece.piece_type == chess.QUEEN:
                attackers = board.attackers(not board.turn, square)
                if len(attackers)>0:
                    unprotected_queens.append(chess.square_name(square))

    return {
        "trapped_rooks": trapped_rooks,
        "hanging_pieces": hanging_pieces,
        "long_diagonal_bishops": long_diagonal_bishops,
        "weak_queens_protection": unprotected_queens,
        "weak_queens": weak_queens
    }
def is_passed_pawn(board, square):
    """
    Check if a pawn on the given square is a passed pawn.
    """
    color = board.piece_at(square).color
    file = chess.square_file(square)

    # Check if there are no opposing pawns on the same or adjacent files
    for adjacent_file in [file - 1, file, file + 1]:
        if 0 <= adjacent_file <= 7:
            adjacent_square = chess.square(adjacent_file, square // 8)
            adjacent_piece = board.piece_at(adjacent_square)

            if (
                adjacent_piece is not None
                and adjacent_piece.piece_type == chess.PAWN
                and adjacent_piece.color != color
            ):
                return False

    return True

def find_passed_pawns(board, color):
    """
    Find and return a list of squares occupied by passed pawns of the given color.
    """
    passed_pawns = []
    squares = []
    for square in chess.SQUARES:
        if board.piece_at(square) == None: continue
        if board.piece_at(square).color == color and board.piece_at(square).piece_type == chess.PAWN: squares.append(square)
    for square in squares:
        if is_passed_pawn(board, square):
            passed_pawns.append(square)

    return passed_pawns

def count_forks(board: chess.Board, color: chess.COLORS):
    forks = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None and piece.color == color:
            for move in board.legal_moves:
                if board.is_capture(move) and move.to_square == square:
                    forks += 1
                    break  # Count each square only once as a fork
    return forks
def is_scholars_mate(board):
    # Check for Scholar's Mate
    return (
        board.piece_at(chess.E2) == chess.Piece(chess.PAWN, board.turn) and
        board.piece_at(chess.F3) == chess.Piece(chess.PAWN, board.turn) and
        board.piece_at(chess.G4) == chess.Piece(chess.QUEEN, board.turn)
    )

def is_fools_mate(board):
    # Check for Fool's Mate
    return (
        board.piece_at(chess.F2) == chess.Piece(chess.PAWN, board.turn) and
        board.piece_at(chess.G2) == chess.Piece(chess.PAWN, board.turn) and
        board.piece_at(chess.H2) == chess.Piece(chess.QUEEN, board.turn)
    )
def Evaluate(board)->float:
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }

    total_evaluation = 0

    # Evaluate piece values
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            piece_value = piece_values[piece.piece_type]
            if piece.color == chess.WHITE:
                total_evaluation += piece_value
            else:
                total_evaluation -= piece_value

    # Evaluate king safety
    white_king_sq = board.king(chess.WHITE)
    black_king_sq = board.king(chess.BLACK)
    
    white_king_safety = len(board.attackers(chess.WHITE, white_king_sq))
    black_king_safety = len(board.attackers(chess.BLACK, black_king_sq))
    
    total_evaluation += (black_king_safety - white_king_safety) * 0.1

    # Evaluate pawn structure
    white_pawns = board.pieces(chess.PAWN, chess.WHITE)
    black_pawns = board.pieces(chess.PAWN, chess.BLACK)
    
    white_pawn_structure = sum(1 for file in chess.FILE_NAMES if len([sq for sq in white_pawns if chess.square_file(sq) == file]) > 1)
    black_pawn_structure = sum(1 for file in chess.FILE_NAMES if len([sq for sq in black_pawns if chess.square_file(sq) == file]) > 1)
    
    total_evaluation += (white_pawn_structure - black_pawn_structure) * 0.2

    # Evaluate checkmates
    if board.is_checkmate():
        total_evaluation += Mate(0).score(mate_score=1000000)/100  # A significant bonus for checkmate
    elif board.is_check():
        is_captured = False
        for x in board.legal_moves:
            if board.is_capture(x):
                total_evaluation -= piece_values[board.piece_type_at(x)-1]
                is_captured = True
                break
        if not is_captured: total_evaluation += 0.5  # A smaller bonus for being in check

    # Evaluate weak castling structures
    if board.has_queenside_castling_rights(chess.WHITE) and board.piece_at(chess.A1).piece_type == chess.ROOK:
        total_evaluation -= 0.5
    if board.has_kingside_castling_rights(chess.WHITE) and board.piece_at(chess.H1).piece_type == chess.ROOK:
        total_evaluation -= 0.5
    if board.has_queenside_castling_rights(chess.BLACK) and board.piece_at(chess.A8).piece_type == chess.ROOK:
        total_evaluation += 0.5
    if board.has_kingside_castling_rights(chess.BLACK) and board.piece_at(chess.H8).piece_type == chess.ROOK:
        total_evaluation += 0.5

    # Evaluate open files with rooks or queens
    for file in chess.FILE_NAMES:
        white_rook_or_queen = board.piece_at(chess.parse_square(file + '1'))
        if white_rook_or_queen and white_rook_or_queen.piece_type in {chess.ROOK, chess.QUEEN}:
            total_evaluation += 0.2
        black_rook_or_queen = board.piece_at(chess.parse_square(file + '8'))
        if black_rook_or_queen and black_rook_or_queen.piece_type in {chess.ROOK, chess.QUEEN}:
            total_evaluation -= 0.2

    # Evaluate forks
    forks = count_forks(board, board.turn)
    analyze = analyze_chess_position(board)
    total_evaluation += forks * 0.5
    for x in find_passed_pawns(board, board.turn):
        total_evaluation += PASSED_RANK[chess.square_rank(x)]/10
    total_evaluation += len(analyze["trapped_rooks"]) * TrappedRook/10
    total_evaluation += len(analyze["hanging_pieces"]) * Hanging/10
    total_evaluation += len(analyze["long_diagonal_bishops"]) * LongDiagonalBishop/10
    total_evaluation += len(analyze["weak_queens_protection"]) * WeakQueenProtection/10
    total_evaluation += len(analyze["weak_queens"]) * WeakQueen/10
    
    # Center control
    center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
    for square in center_squares:
        piece = board.piece_at(square)
        if piece is not None:
            if piece.color == board.turn:
                total_evaluation += 1
            else:
                total_evaluation -= 1
    return round(total_evaluation * pow(-1, int(not board.turn))*10, 3)

"""
def minimax(board, depth, maximizing_player, alpha, beta, allmove: str = ""):
    if depth == 0 or board.is_game_over():
        return Evaluate(board)

    legal_moves = list(board.legal_moves)

    if maximizing_player:
        max_eval = float('-inf')
        for move in legal_moves:
            san = board.san(move)
            board.push(move)
            eval_score = minimax(board, depth - 1, False, alpha, beta, allmove + " " + san)
            transposition.append({"FEN": board.fen(), "score": eval_score})
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            board.pop()
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in legal_moves:
            san = board.san(move)
            board.push(move)
            eval_score = minimax(board, depth - 1, True, alpha, beta, allmove + " " + san)
            transposition.append({"FEN": board.fen(), "score": eval_score})
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            board.pop()
            if beta <= alpha:
                break
        return min_eval
"""
def minimax(board, depth, void, alpha, beta, start_time, allmove: str = "", nodes: int = 1, capturedepth=5):
    if depth == 0 or board.is_game_over() or board.is_fivefold_repetition() or board.is_seventyfive_moves() or board.is_fifty_moves() or board.is_repetition() or (board.is_insufficient_material() and len(list(board.generate_legal_captures())) != 0):
        return minimaxcaptures(board, capturedepth, void, alpha, beta, start_time, allmove, nodes + 1)
    elif depth == 0 or board.is_game_over() or board.is_fivefold_repetition() or board.is_seventyfive_moves() or board.is_fifty_moves() or board.is_repetition() or (board.is_insufficient_material() and len(list(board.generate_legal_captures())) == 0):
        return [Evaluate(board), nodes]
    
    current_time = time.time()
    elapsed_time = current_time - start_time
    if elapsed_time * 1000 >= int(movetime):
        return [Evaluate(board), nodes]

    legal_moves = list(board.generate_legal_moves())
    eval_score = None
    max_eval = float('999999.99')

    for move in legal_moves:
        san = board.san(move)
        fen = board.fen()
        board.push(move)

        # Search for the element in the array
        for item in transposition:
            if item["FEN"] == fen:
                eval_score = [item["score"], nodes + 1]
                break

        if eval_score is None:
            eval_score = minimaxcaptures(board, capturedepth, void, alpha, beta, start_time, allmove + " " + san, nodes + 1)
            if eval_score == [0, 0]:
                eval_score = minimax(board, depth - 1, void, alpha, beta, start_time, allmove + " " + san, nodes + 1, capturedepth)
            print("info depth %i move %s score cp %s time %i" % (depth, allmove + " " + san, eval_score[0] * 100, elapsed_time))
            if elapsed_time >= int(movetime):
                transposition.append({"FEN": board.fen(), "score": eval_score[0], "nodes": eval_score[1]})
                board.pop()
                return [Evaluate(board), nodes]

            transposition.append({"FEN": board.fen(), "score": eval_score[0], "nodes": eval_score[1]})

        max_eval = max(max_eval, eval_score[0])
        alpha = max(alpha, eval_score[0])
        board.pop()

        if beta <= alpha:
            break

        nodes += 1
        eval_score = None

    return [max_eval, nodes]

def minimaxcaptures(board, depth, void, alpha, beta, start_time, allmove: str = "", nodes: int = 1):
    if depth == 0 or board.is_game_over() or board.is_fivefold_repetition() or board.is_seventyfive_moves() or board.is_fifty_moves() or board.is_repetition() or board.is_insufficient_material() or len(list(board.generate_legal_captures())) == 0:
        return [Evaluate(board), nodes]

    current_time = time.time()
    elapsed_time = current_time - start_time
    if elapsed_time * 1000 >= movetime:
        return [Evaluate(board), nodes]

    legal_moves = list(board.generate_legal_captures())
    eval_score = None
    max_eval = float('999999.99')

    for move in legal_moves:
        san = board.san(move)
        fen = board.fen()
        board.push(move)

        # Search for the element in the array
        for item in transposition:
            if item["FEN"] == fen:
                eval_score = [item["score"], nodes + 1]
                break

        if eval_score is None:
            eval_score = minimaxcaptures(board, depth - 1, void, alpha, beta, start_time, allmove + " " + san, nodes + 1)
            print("info depth %i move %s score cp %s time %i" % (depth, allmove + " " + san, eval_score[0] * 100, elapsed_time))
            if elapsed_time * 1000 >= int(movetime):
                transposition.append({"FEN": board.fen(), "score": eval_score[0], "nodes": eval_score[1]})
                board.pop()
                return [Evaluate(board), nodes]

            transposition.append({"FEN": board.fen(), "score": eval_score[0], "nodes": eval_score[1]})

        max_eval = max(max_eval, eval_score[0])
        alpha = max(alpha, eval_score[0])
        board.pop()

        if beta <= alpha:
            break

        nodes += 1
        eval_score = None

    return [max_eval, nodes]

def selectionSort(array, array2, size):
    
    for ind in range(size):
        min_index = ind
 
        for j in range(ind + 1, size):
            # select the minimum element in every iteration
            if array[j] < array[min_index]:
                min_index = j
         # swapping the elements to sort the array
        (array[ind], array[min_index]) = (array[min_index], array[ind])
        (array2[ind], array2[min_index]) = (array2[min_index], array2[ind])
def best_move(board, depth, movetime2: int):
    start_time = int(round(time.time() * 1000))
    legal_moves = list(board.legal_moves)
    best_move = None
    best_eval = float('-inf')
    best_moves = []
    mate = []
    movetime = movetime2
    for move in legal_moves:
        current_time = int(round(time.time() * 1000))
        if current_time - start_time >= movetime2:
            selectionSort(mate, best_moves, len(mate) - 1)
            return best_moves[0]

        san = board.san(move)
        board.push(move)
        eval_score = minimax(board, depth, False, float('-inf'), float('inf'), movetime, san, 1, depth)
        board.pop()

        if eval_score[0] >= best_eval:
            best_eval = eval_score[0]
            best_move = move
            mate.append(eval_score[1])
            best_moves.append(move)

    selectionSort(mate, best_moves, len(mate) - 1)
    return best_moves[0]
"""
def minimax(board, depth: int, move: str, alpha: int, beta: int):
    if depth == 0 or (board.is_game_over() and not board.is_checkmate()):
        return evaluation(board)
    print("info depth", depth)
    if board.is_checkmate():
        return -1000000
    maxEval = -1000000

    # Loop through promotion moves
    for currmove in board.legal_moves:
        if currmove.promotion is not None:
            san = board.san(currmove)
            board.push(currmove)
            eval = -minimax(board, depth - 1, move + " " + san, -alpha, -beta)
            print("info depth %i move %s score %i" % (depth, move + " " + san, eval))
            transposition.append({"FEN": board.fen(), "score": eval})
            board.pop()
            maxEval = max(maxEval, eval)

    # Loop through non-promotion moves
    for currmove in board.legal_moves:
        if currmove.promotion is None:
            san = board.san(currmove)
            board.push(currmove)
            eval = -minimax(board, depth - 1, move + " " + san, -alpha, -beta)
            print("info depth %i move %s score %i" % (depth, move + " " + san, eval))
            transposition.append({"FEN": board.fen(), "score": eval})
            board.pop()
            maxEval = max(maxEval, eval)
    return maxEval
"""
"""
Client.request_config['headers']['User-Agent'] = 'WinHTTP'
dailypuzzle=get_current_daily_puzzle().puzzle
print("Title:", dailypuzzle.title)
print("Time:", datetime.fromtimestamp(dailypuzzle.publish_time))
print("FEN:", dailypuzzle.fen)
"""
transposition = []
# board = chess.Board(dailypuzzle.fen)
board = chess.Board()
try:
    with open("transposition.txt", "r") as file:
        x = file.readline()
        while x!='':
            transposition.append(eval(x.strip().replace("'score': -inf", "'score': float('-inf')").replace("'score': inf", "'score': float('inf')")))
            x = file.readline()
except:
    open("transposition.txt", "w").close()
print("Python-ChessEngine")
userinput = input()
while userinput != "quit":
    tokens = [userinput.split(" ")[x + 1] for x in range(len(userinput.split(" "))-1)]
    if "go" in userinput:
        if "depth" in userinput:
            if "movetime" in tokens: movetime = tokens[tokens.index("movetime")+1]
            if "wtime" in tokens and board.turn == chess.WHITE: movetime = tokens[tokens.index("wtime")+1]
            if "btime" in tokens and board.turn == chess.BLACK: movetime = tokens[tokens.index("wtime")+1]
            else: movetime = float('600000')
            bestMove = best_move(board, int(tokens[tokens.index("depth") + 1]), int(movetime))
            print("bestmove %s" % (bestMove))
    if userinput=="go":
        movetime = float('600000')
        bestMove = best_move(board, 249, int(movetime))
        print("bestmove %s" % (bestMove))
    if "position" in userinput:
        if tokens[0] == "startpos":
            board.set_fen(chess.STARTING_FEN)
        if "moves" in tokens and "fen" not in tokens:
            moves = [userinput.split(" ")[x + 1] for x in range(2, len(userinput.split(" "))-1)]
            for move in moves:
                board.push_uci(move)
        if "fen" in tokens:
            fen = tokens[tokens.index("fen")+1]
            if "moves" in tokens:
                moves = [tokens[x + 1] for x in range(tokens.index("fen") + 1 + 2, len(userinput.split(" "))-1)]
                for move in moves:
                    board.push_uci(move)
    if userinput == "uci":
        print("id name Python-ChessEngine")
        print("id author winapiadmin")
        print("\nuciok")
    if "setoption" in userinput: pass
    if userinput == "ucinewgame": pass
    if userinput == "isready": print("readyok")
    userinput = input()
with open("transposition.txt", "a") as file:
    for x in transposition:
        file.write(str(x).replace("'score': -inf", "'score': float('-inf')").replace("'score': inf", "'score': float('inf')") + "\n")
