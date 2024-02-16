print("Python-ChessEngine")
print("Idealed by stackoverflow")
import chess
pawnValue = 100
knightValue = 300
bishopValue = 320
rookValue = 500
queenValue = 900
kingValue = 20_000

mateValue = 100_000_000_000-1

BISHOP_PAIR_BONNUS = 0.5

CASTLE_BONNUS = 100
GOOD_CASTLE_BONNUS = 300
OPEN_FILE_CASTLE_MALUS = 250
KING_ON_OPEN_FILE_MALUS = 100
FIANCHETTO_BONNUS = 50

SHORT_CASTLE = (chess.G1, chess.G2, chess.H1, chess.H2, chess.G8, chess.G7, chess.H8, chess.H7)
LONG_CASTLE = (chess.A1, chess.B1, chess.C1, chess.A2, chess.B2, chess.C2,
               chess.A8, chess.B8, chess.C8, chess.A7, chess.B7, chess.C7)
SHORT_PAWN_PROTECTION_W = (chess.F2, chess.G2, chess.H2,
                           chess.F3, chess.G3, chess.H3)
SHORT_PAWN_PROTECTION_B = (chess.F8, chess.G8, chess.H8,
                           chess.F7, chess.G7, chess.H7)
LONG_PAWN_PROTECTION_W = (chess.A2, chess.B2, chess.C2,
                          chess.A3, chess.B3, chess.C3)
LONG_PAWN_PROTECTION_B = (chess.A7, chess.B7, chess.C7,
                          chess.A8, chess.B8, chess.C8)


K = [-30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20]

K_end = [-50,-40,-30,-20,-20,-30,-40,-50,
        -30,-20,-10,  0,  0,-10,-20,-30,
        -30,-10, 20, 30, 30, 20,-10,-30,
        -30,-10, 30, 40, 40, 30,-10,-30,
        -30,-10, 30, 40, 40, 30,-10,-30,
        -30,-10, 20, 30, 30, 20,-10,-30,
        -30,-30,  0,  0,  0,  0,-30,-30,
        -50,-30,-30,-30,-30,-30,-30,-50]

Q = [-20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20]

R = [ 0,  0,  0,  0,  0,  0,  0,  0,
      5, 10, 10, 10, 10, 10, 10,  5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
      0,  0,  0,  5,  5,  0,  0,  0]

N = [-50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50]

B = [20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20]

P = [ 0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5,  5, 10, 25, 25, 10,  5,  5,
    0,  0,  0, 20, 20,  0,  0,  0,
    5, -5,-10,  0,  0,-10, -5,  5,
    5, 10, 10,-20,-20, 10, 10,  5,
    0,  0,  0,  0,  0,  0,  0,  0]

pqst_table = {chess.KING: [K, K_end],
            chess.QUEEN: Q,
            chess.ROOK: R,
            chess.KNIGHT: N,
            chess.BISHOP: B,
            chess.PAWN: P}


def popcount(a: int) -> int :
    return bin(a).count("1")

def scale_to_white_view(board: chess.Board, eval: float) -> float :
    perspective = 1 if board.turn == chess.WHITE else -1
    return eval * perspective

def evaluate(board: chess.Board=chess.Board()) -> float :

    if len(list(board.legal_moves)) == 0 :
        if board.is_check() :
            return -mateValue + 1 # +1 pour la recherche de mats courts
        return 0

    whiteEval = 0
    blackEval = 0

    endgameWeight = game_phase(board)
    pqst_evalu = pqst_eval(board, endgameWeight)

    whiteEval += countMaterial(board, chess.WHITE)
    blackEval += countMaterial(board, chess.BLACK)

    whiteEval += pqst_evalu[0]
    blackEval += pqst_evalu[1]

    mob = mobility(board)
    whiteEval += mob[0]
    blackEval += mob[1]

    whiteEval += bishop_pair(board, chess.WHITE)
    blackEval += bishop_pair(board, chess.BLACK)

    whiteEval += king_safety(board, chess.WHITE, endgameWeight) * 0.3
    blackEval += king_safety(board, chess.BLACK, endgameWeight) * 0.3

    evaluation = whiteEval-blackEval

    perspective = 1 if board.turn == chess.WHITE else -1

    return evaluation * perspective

def countMaterial(board: chess.Board=chess.Board(), color: chess.Color=chess.WHITE) -> float :
    material = 0
    material += len(board.pieces(chess.PAWN, color)) * pawnValue
    material += len(board.pieces(chess.KNIGHT, color)) * knightValue
    material += len(board.pieces(chess.BISHOP, color)) * bishopValue
    material += len(board.pieces(chess.ROOK, color)) * rookValue
    material += len(board.pieces(chess.QUEEN, color)) * queenValue
    return material

def pqst_eval(board: chess.Board, endgame_value: int) -> tuple :
    white_eval = 0
    black_eval = 0
    for square in chess.SQUARES :
        piece = board.piece_at(square)
        if piece != None :
            if piece.color :
                if piece.piece_type == chess.KING :
                    white_eval += ((pqst_table[piece.piece_type][0][63 - square] * (256 - endgame_value)) + (pqst_table[piece.piece_type][1][63 - square] * endgame_value)) / 256
                else :
                    white_eval += pqst_table[piece.piece_type][63 - square]
            else :
                if piece.piece_type == chess.KING :
                    black_eval += ((pqst_table[piece.piece_type][0][square] * (256 - endgame_value)) + (pqst_table[piece.piece_type][1][square] * endgame_value)) / 256
                else :
                    black_eval += pqst_table[piece.piece_type][square]
    return white_eval, black_eval

def game_phase(board: chess.Board) -> int :
    '''De 0.5 (ouv) à 256.5 (endg)'''
    PawnPhase = 0
    KnightPhase = 1
    BishopPhase = 1
    RookPhase = 2
    QueenPhase = 4
    TotalPhase = PawnPhase*16 + KnightPhase*4 + BishopPhase*4 + RookPhase*4 + QueenPhase*2

    phase = TotalPhase

    phase -= popcount(board.pawns) * PawnPhase
    phase -= popcount(board.knights) * KnightPhase
    phase -= popcount(board.bishops) * KnightPhase
    phase -= popcount(board.rooks) * RookPhase
    phase -= popcount(board.queens) * QueenPhase

    phase = (phase * 256 + (TotalPhase / 2)) / TotalPhase
    return phase

def mobility(board: chess.Board) -> int :

    w_mob = 0
    b_mob = 0
    occupied = board.occupied

    for square in chess.SQUARES :

        piece = board.piece_at(square)
        if piece == None :
            continue

        if piece.color :
            w_mob += mobility_square(square, piece.piece_type, occupied)
        else :
            b_mob += mobility_square(square, piece.piece_type, occupied)

    return w_mob, b_mob


def mobility_square(square: chess.Square, piece: chess.Piece, occupied: int) -> int: 

    if piece == None : return 0

    if piece == chess.ROOK :
        return popcount((chess.BB_RANK_ATTACKS[square][chess.BB_RANK_MASKS[square] & occupied] | chess.BB_FILE_ATTACKS[square][chess.BB_FILE_MASKS[square] & occupied]) & ~occupied)

    if piece == chess.KNIGHT :
        return popcount(chess.BB_KNIGHT_ATTACKS[square]  & ~occupied )

    if piece == chess.BISHOP :
        return popcount(chess.BB_DIAG_ATTACKS[square][chess.BB_DIAG_MASKS[square] & occupied] & ~occupied)

    if piece == chess.QUEEN :
        return popcount((chess.BB_RANK_ATTACKS[square][chess.BB_RANK_MASKS[square] & occupied] |
          chess.BB_FILE_ATTACKS[square][chess.BB_FILE_MASKS[square] & occupied] |
          chess.BB_DIAG_ATTACKS[square][chess.BB_DIAG_MASKS[square] & occupied]) & ~occupied)

    return 0


def bishop_pair(board: chess.Board, color: chess.Color) :
    if len(board.pieces(chess.BISHOP, color)) == 2 :
        return BISHOP_PAIR_BONNUS
    return 0


def king_safety(board: chess.Board, color: chess.Color, endgame_weight) :

    # TODO : pawn storm
    
    score = 0
    king = board.king(color)
    eg_factor = (256.5- endgame_weight)/256.5

    if color : # Si Blancs
        # Test petit roque
        if king in SHORT_CASTLE :
            score += CASTLE_BONNUS

            if bool(board.pawns & 0x700) & bool(board.pawns & 0x10600) & bool(board.pawns & 0x20500) : # Si tous les pions protègent le roque
                score += GOOD_CASTLE_BONNUS

            else :
                # max pour éviter que tous les pions se mettent devant le roi
                score -= OPEN_FILE_CASTLE_MALUS * (max((1-popcount(board.pawns & 0x40400)) + (1-popcount(board.pawns & 0x20200)) + (1-popcount(board.pawns & 0x10100)), 0))
   
            if (board.bishops & 0x200) : # Fianchetto
                    score += FIANCHETTO_BONNUS

        # Test grand roque
        elif king in LONG_CASTLE :
            score += CASTLE_BONNUS

            if bool(board.pawns & 0xe000) & bool(board.pawns & 0x40a000) & bool(board.pawns & 0x806000) : # Si tous les pions protègent le roque
                score += GOOD_CASTLE_BONNUS
            else :
                # max pour éviter que tous les pions se mettent devant le roi
                score -= OPEN_FILE_CASTLE_MALUS * (max((1-popcount(board.pawns & 0x808000)) + (1-popcount(board.pawns & 0x404000)) + (1-popcount(board.pawns & 0x202000)), 0))

            if (board.bishops & 0x4000) : # Fianchetto
                    score += FIANCHETTO_BONNUS

        # Roi non-roqué ou déroqué
        else :
            file = chess.square_file(king)
            my_pawn_type = chess.Piece(chess.PAWN, color)
            for square in chess.SQUARES :
                if chess.square_file(square) == file :
                    piece = board.piece_at(square)
                    if piece == my_pawn_type :
                        score += KING_ON_OPEN_FILE_MALUS * eg_factor
                        break
            score -= KING_ON_OPEN_FILE_MALUS * eg_factor

    elif not color : # Si Noirs
        # Test petit roque
        if king in SHORT_CASTLE :
            score += CASTLE_BONNUS

            if bool(board.pawns & 0x7000000000000) & bool(board.pawns & 0x5020000000000) & bool(board.pawns & 0x6010000000000) : # Si tous les pions protègent le roque
                score += GOOD_CASTLE_BONNUS
            else :
                # max pour éviter que tous les pions se mettent devant le roi
                score -= OPEN_FILE_CASTLE_MALUS * (max((1-popcount(board.pawns & 0x4040000000000)) + (1-popcount(board.pawns & 0x2020000000000)) + (1-popcount(board.pawns & 0x1010000000000)), 0))

            if (board.bishops & 0x2000000000000) : # Fianchetto
                        score += FIANCHETTO_BONNUS

        # Test grand roque
        elif king in LONG_CASTLE :
            score += CASTLE_BONNUS

            if bool(board.pawns & 0xe0000000000000) & bool(board.pawns & 0x60800000000000) & bool(board.pawns & 0xa0400000000000) : # Si tous les pions protègent le roque
                score += GOOD_CASTLE_BONNUS
            else :
                # max pour éviter que tous les pions se mettent devant le roi
                score -= OPEN_FILE_CASTLE_MALUS * (max((1-popcount(board.pawns & 0x80800000000000)) + (1-popcount(board.pawns & 0x40400000000000)) + (1-popcount(board.pawns & 0x20200000000000)), 0))

            if (board.bishops & 0x40000000000000) : # Fianchetto
                    score += FIANCHETTO_BONNUS

        # Roi non-roqué ou déroqué
        else :
            file = chess.square_file(king)
            my_pawn_type = chess.Piece(chess.PAWN, color)
            for square in chess.SQUARES :
                if chess.square_file(square) == file :
                    piece = board.piece_at(square)
                    if piece == my_pawn_type :
                        score += KING_ON_OPEN_FILE_MALUS * eg_factor
                        break
            score -= KING_ON_OPEN_FILE_MALUS * eg_factor

    
    return score

def minimax(board, depth, void, alpha, beta, allmove: str = "", nodes: int = 1, capturedepth=5):
    if depth == 0:
        print("info depth %i move (none) score cp %s" % (depth, int(scale_to_white_view(board, evaluate(board)))))
        return [int(scale_to_white_view(board, evaluate(board))), nodes]

    legal_moves = list(board.generate_legal_moves())
    eval_score = None
    max_eval = 100_000_000_000

    for move in legal_moves:
        san = str(move)
        fen = board.fen()
        board.push(move)

        # Search for the element in the array
        for item in transposition:
            if item["FEN"] == fen:
                eval_score = [item["score"], nodes + 1]
                break

        if eval_score is None:
            eval_score = minimax(board, depth - 1, void, alpha, beta, allmove + " " + san, nodes + 1, capturedepth)
            transposition.append({"FEN": board.fen(), "score": eval_score[0], "nodes": eval_score[1]})
        print("info depth %i move %s score cp %s" % (depth, allmove + " " + san, eval_score[0]))
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
def best_move(board, depth):
    legal_moves = list(board.legal_moves)
    best_move = None
    best_eval = 100_000_000_000
    best_moves = []
    mate = []
    if board.fen() == chess.STARTING_FEN: legal_moves = [chess.Move.from_uci("e2e4"), chess.Move.from_uci("d2d4"), chess.Move.from_uci("g1f3"), chess.Move.from_uci("b1c3")]
    for move in legal_moves:
        san = str(move)
        board.push(move)
        eval_score = minimax(board, depth, False, 100_000_000_000, 100_000_000_000, san, 1, depth)
        board.pop()

        if eval_score[0] >= best_eval:
            best_eval = eval_score[0]
            best_move = move
            mate.append(eval_score[1])
            best_moves.append(move)

    selectionSort(mate, best_moves, len(mate) - 1)
    return best_moves[0]

transposition = []
board = chess.Board()
try:
    with open("transposition.txt", "r") as file:
        x = file.readline()
        while x!='':
            transposition.append(eval(x.strip()))
            x = file.readline()
except:
    open("transposition.txt", "w").close()
userinput = input()
while userinput != "quit":
    tokens = [userinput.split(" ")[x + 1] for x in range(len(userinput.split(" "))-1)]
    if userinput.strip()=="go":
        bestMove = best_move(board, 249)
        print("bestmove %s" % (bestMove))
    elif "go" in userinput:
        if "depth" in userinput:
            bestMove = best_move(board, int(tokens[tokens.index("depth") + 1]))
            print("bestmove %s" % (bestMove))
    elif "position" in userinput:
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
    elif userinput == "uci":
        print("id name Python-ChessEngine")
        print("id author winapiadmin")
        print("\nuciok")
    elif "setoption" in userinput: pass
    elif userinput == "ucinewgame":
        transposition = []
    elif userinput == "isready": print("readyok")
    elif userinput == "stop": pass
    elif userinput == "d":
        print("+---+---+---+---+---+---+---+---+")
        for row in range(8):
            for col in range(8):
                if board.piece_at(chess.square(col, row))!=None:
                    print("| %s" % (board.piece_at(chess.square(col, row))), end = ' ')
                else:
                    print("|  ", end = ' ')
            print("|", row)
            print("+---+---+---+---+---+---+---+---+")
        print()
        print("Fen:", board.fen())
        print("Checkers:", end = ' ')
        for y in board.checkers():
            print(FILE_NAMES[square_file(y)]+RANK_NAMES[square_rank(y)], end = ' ')
        print()
    else:
        print("Command `%s` not implemented" % (userinput))
    userinput = input()
with open("transposition.txt", "a") as file:
    for x in transposition:
        file.write(str(x) + "\n")
