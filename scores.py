    
def S(mg, eg):
    return int((eg >> 16)) + mg

# KingProtector[knight/bishop] contains penalty for each distance unit to own king
KING_PROTECTOR = [S(9, 9), S(7, 9)]

# Outpost[knight/bishop] contains bonuses for each knight or bishop occupying a
# pawn protected square on rank 4 to 6 which is also safe from a pawn attack.
OUTPOST = [S(54, 34), S(31, 25)]

# PassedRank[Rank] contains a bonus according to the rank of a passed pawn
PASSED_RANK = [S(0, 0), S(2, 38), S(15, 36), S(22, 50), S(64, 81), S(166, 184), S(284, 269)] + [0 for _ in range(1)]

RookOnClosedFile = S(10, 5)
ROOK_ON_OPEN_FILE = [S(18, 8), S(49, 26)]

# ThreatByMinor/ByRook[attacked PieceType] contains bonuses according to
# which piece type attacks which one. Attacks on lesser pieces which are
# pawn-defended are not considered.
THREAT_BY_MINOR = [S(0, 0), S(6, 37), S(64, 50), S(82, 57), S(103, 130), S(81, 163)] + [0 for _ in range(2)]

THREAT_BY_ROOK = [S(0, 0), S(3, 44), S(36, 71), S(44, 59), S(0, 39), S(60, 39)] + [0 for _ in range(2)]

# Assorted bonuses and penalties
UncontestedOutpost = S(0, 10)
BishopOnKingRing = S(24, 0)
BishopXRayPawns = S(4, 5)
FlankAttacks = S(8, 0)
Hanging = S(72, 40)
KnightOnQueen = S(16, 11)
LongDiagonalBishop = S(45, 0)
MinorBehindPawn = S(18, 3)
PassedFile = S(13, 8)
PawnlessFlank = S(19, 97)
ReachableOutpost = S(33, 19)
RestrictedPiece = S(6, 7)
RookOnKingRing = S(16, 0)
SliderOnQueen = S(62, 21)
ThreatByKing = S(24, 87)
ThreatByPawnPush = S(48, 39)
ThreatBySafePawn = S(167, 99)
TrappedRook = S(55, 13)
WeakQueenProtection = S(14, 0)
WeakQueen = S(57, 19)
