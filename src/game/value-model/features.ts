import { bombardedSquares } from "@/game/move-logic";
import {
  Board,
  Coordinate,
  Player,
  ReserveFleet,
  UnitType,
  Units,
} from "@/game/engine";
import { unitScores } from "@/game/eval";

const PLAYERS: Player[] = ["RED", "BLUE"];
const PIECE_TYPES = [
  "HQ",
  "INFANTRY",
  "ARMORED_INFANTRY",
  "AIRBORNE_INFANTRY",
  "ARTILLERY",
  "ARMORED_ARTILLERY",
  "HEAVY_ARTILLERY",
] as const satisfies readonly UnitType[];
const RESERVE_TYPES = [
  "INFANTRY",
  "ARMORED_INFANTRY",
  "AIRBORNE_INFANTRY",
  "ARTILLERY",
  "ARMORED_ARTILLERY",
  "HEAVY_ARTILLERY",
] as const satisfies readonly (keyof ReserveFleet)[];
const INFANTRY_TYPES = new Set<UnitType>([
  "INFANTRY",
  "ARMORED_INFANTRY",
  "AIRBORNE_INFANTRY",
]);
const ARTILLERY_TYPES = new Set<UnitType>([
  "ARTILLERY",
  "ARMORED_ARTILLERY",
  "HEAVY_ARTILLERY",
]);

const SIDE_FEATURE_NAMES = [
  ...PIECE_TYPES.map((type) => `board_${type.toLowerCase()}`),
  ...RESERVE_TYPES.map((type) => `reserve_${type.toLowerCase()}`),
  "material_board",
  "material_total",
  "pieces_board",
  "pieces_reserve",
  "infantry_board",
  "artillery_board",
  "home_rank_occupancy",
  "relocation_options",
  "mean_relocation_options",
  "immobile_count",
  "home_rank_immobile_count",
  "connected_components",
  "largest_component_ratio",
  "max_rank_occupancy",
  "advancement_mean",
  "advancement_max",
  "unsupported_count",
  "unsupported_value",
  "support_distance_mean",
  "overextended_count",
  "overextended_value",
  "artillery_adjacent_pairs",
  "heavy_artillery_centered",
  "artillery_diagonal_count",
  "artillery_cardinal_count",
  "bombarded_squares",
  "bombarded_enemy_count",
  "bombarded_enemy_value",
  "artillery_protected_count",
  "artillery_diagonal_infantry_cover",
  "artillery_cardinal_infantry_cover",
  "infantry_engagements",
  "paratrooper_ready",
  "paratrooper_deployed",
  "paratrooper_distance_home",
  "paratrooper_supported",
  "paratrooper_engaged",
  "hq_bombarded",
  "hq_adjacent_enemy_infantry",
  "hq_adjacent_friendly",
  "hq_escape_squares",
  "pseudo_mobility",
] as const;

/**
 * Formation features added for the next challenger. Keep these separate from
 * SIDE_FEATURE_NAMES so the deployed incumbent retains its exact feature
 * schema and checkpoint compatibility while a v2 challenger is trained.
 */
const STRUCTURE_V2_SIDE_FEATURE_NAMES = [
  "infantry_vertical_adjacent_pairs",
  "infantry_diagonal_adjacent_pairs",
  "infantry_same_file_run_excess",
  "infantry_isolated_count",
  "infantry_distinct_files",
  "infantry_file_span",
  "infantry_rank_span",
  "infantry_frontier_count",
  "material_pair_distance_mean",
  "material_file_span",
  "material_rank_span",
] as const;

/**
 * Tactical HQ-approach evidence for the next append-only checkpoint. The v1
 * model only knows about bombardment, adjacent infantry, and escape squares;
 * it can therefore remain highly confident until an attacker is already at
 * the HQ. These features expose the approach geometry before that last ply.
 */
const TACTICAL_V3_SIDE_FEATURE_NAMES = [
  "hq_enemy_infantry_distance_min",
  "hq_enemy_armored_infantry_distance_min",
  "hq_enemy_airborne_infantry_distance_min",
  "hq_enemy_infantry_within_two",
  "hq_enemy_infantry_within_three",
  "hq_friendly_infantry_within_two",
  "hq_friendly_infantry_within_three",
  "hq_attack_pressure",
  "hq_defense_density",
] as const;

const DIFFERENCE_FEATURES = [
  "material_board",
  "material_total",
  "pieces_board",
  "infantry_board",
  "artillery_board",
  "home_rank_occupancy",
  "relocation_options",
  "immobile_count",
  "home_rank_immobile_count",
  "connected_components",
  "largest_component_ratio",
  "unsupported_value",
  "overextended_value",
  "bombarded_enemy_value",
  "artillery_protected_count",
  "paratrooper_ready",
  "paratrooper_deployed",
  "hq_bombarded",
  "hq_escape_squares",
  "pseudo_mobility",
] as const;

export const VALUE_FEATURE_NAMES = [
  "turn_progress",
  "board_fill",
  "surviving_unit_fraction",
  "own_to_move",
  ...SIDE_FEATURE_NAMES.map((name) => `own_${name}`),
  ...SIDE_FEATURE_NAMES.map((name) => `opp_${name}`),
  ...DIFFERENCE_FEATURES.map((name) => `diff_${name}`),
] as const;

export const VALUE_FEATURE_NAMES_V2 = [
  // Preserve every incumbent index so its exact trees can serve as a fair
  // baseline against v2. New evidence is append-only.
  ...VALUE_FEATURE_NAMES,
  ...STRUCTURE_V2_SIDE_FEATURE_NAMES.map((name) => `own_${name}`),
  ...STRUCTURE_V2_SIDE_FEATURE_NAMES.map((name) => `opp_${name}`),
  ...STRUCTURE_V2_SIDE_FEATURE_NAMES.map((name) => `diff_${name}`),
] as const;

export const VALUE_FEATURE_NAMES_V3 = [
  // Preserve both earlier schemas exactly. Native and TypeScript inference
  // select the extractor from the artifact's declared append-only length.
  ...VALUE_FEATURE_NAMES_V2,
  ...TACTICAL_V3_SIDE_FEATURE_NAMES.map((name) => `own_${name}`),
  ...TACTICAL_V3_SIDE_FEATURE_NAMES.map((name) => `opp_${name}`),
  ...TACTICAL_V3_SIDE_FEATURE_NAMES.map((name) => `diff_${name}`),
] as const;

export type ValueFeatureName = (typeof VALUE_FEATURE_NAMES)[number];
export type ValueFeatureRecord = Record<ValueFeatureName, number>;

export interface ValuePosition {
  board: Board;
  redReserve: ReserveFleet;
  blueReserve: ReserveFleet;
  currentPlayer: Player;
  turnNumber: number;
}

type PlacedPiece = {
  at: Coordinate;
  piece: Exclude<Board[number][number], null>;
};

type SideFeatureName = (typeof SIDE_FEATURE_NAMES)[number];
type StructureV2SideFeatureName =
  (typeof STRUCTURE_V2_SIDE_FEATURE_NAMES)[number];
type SideFeatures = Record<SideFeatureName, number>;
type StructureV2SideFeatures = Record<StructureV2SideFeatureName, number>;
type TacticalV3SideFeatureName =
  (typeof TACTICAL_V3_SIDE_FEATURE_NAMES)[number];
type TacticalV3SideFeatures = Record<TacticalV3SideFeatureName, number>;

function otherPlayer(player: Player): Player {
  return player === "RED" ? "BLUE" : "RED";
}

function homeRank(player: Player): number {
  return player === "RED" ? 7 : 0;
}

function chebyshev(a: Coordinate, b: Coordinate): number {
  return Math.max(Math.abs(a[0] - b[0]), Math.abs(a[1] - b[1]));
}

function manhattan(a: Coordinate, b: Coordinate): number {
  return Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]);
}

function inside(row: number, column: number): boolean {
  return row >= 0 && row < 8 && column >= 0 && column < 8;
}

function neighbors(at: Coordinate, diagonal: boolean): Coordinate[] {
  const result: Coordinate[] = [];
  for (let dr = -1; dr <= 1; dr++) {
    for (let dc = -1; dc <= 1; dc++) {
      if (dr === 0 && dc === 0) continue;
      if (!diagonal && Math.abs(dr) + Math.abs(dc) !== 1) continue;
      const row = at[0] + dr;
      const column = at[1] + dc;
      if (inside(row, column)) result.push([row, column]);
    }
  }
  return result;
}

function piecesFor(board: Board, player: Player): PlacedPiece[] {
  const result: PlacedPiece[] = [];
  board.forEach((rank, row) =>
    rank.forEach((piece, column) => {
      if (piece?.player === player) result.push({ at: [row, column], piece });
    })
  );
  return result;
}

function reserveFor(position: ValuePosition, player: Player): ReserveFleet {
  return player === "RED" ? position.redReserve : position.blueReserve;
}

function nearestSupportDistance(
  candidate: PlacedPiece,
  friendly: PlacedPiece[]
): number {
  const distances = friendly
    .filter(
      (other) =>
        other !== candidate &&
        other.piece.type !== "HQ" &&
        other.piece.type !== "AIRBORNE_INFANTRY"
    )
    .map((other) => chebyshev(candidate.at, other.at));
  return distances.length === 0 ? 8 : Math.min(...distances);
}

function isHeavyCentered(
  heavy: PlacedPiece,
  artillery: PlacedPiece[]
): boolean {
  const [row, column] = heavy.at;
  const sameRank = artillery.filter(
    (gun) => gun !== heavy && gun.at[0] === row
  );
  const sameFile = artillery.filter(
    (gun) => gun !== heavy && gun.at[1] === column
  );
  return (
    (sameRank.some((gun) => gun.at[1] < column) &&
      sameRank.some((gun) => gun.at[1] > column)) ||
    (sameFile.some((gun) => gun.at[0] < row) &&
      sameFile.some((gun) => gun.at[0] > row))
  );
}

function pseudoMobility(board: Board, pieces: PlacedPiece[]): number {
  let moves = 0;
  for (const { at, piece } of pieces) {
    if (piece.type === "HQ") continue;
    const speed = Units[piece.type].mobility;
    for (
      let row = Math.max(0, at[0] - speed);
      row <= Math.min(7, at[0] + speed);
      row++
    ) {
      for (
        let column = Math.max(0, at[1] - speed);
        column <= Math.min(7, at[1] + speed);
        column++
      ) {
        if (row === at[0] && column === at[1]) continue;
        if (
          chebyshev(at, [row, column]) <= speed &&
          board[row][column] === null
        ) {
          moves++;
        }
      }
    }
  }
  return Math.log1p(moves);
}

function structureAndOptionMetrics(
  board: Board,
  pieces: PlacedPiece[],
  ownHomeRank: number
): {
  relocationOptions: number;
  meanRelocationOptions: number;
  immobileCount: number;
  homeRankImmobileCount: number;
  connectedComponents: number;
  largestComponentRatio: number;
} {
  const material = pieces.filter(
    ({ piece }) => piece.type !== "HQ" && piece.type !== "AIRBORNE_INFANTRY"
  );
  let relocationOptions = 0;
  let immobileCount = 0;
  let homeRankImmobileCount = 0;
  for (const candidate of material) {
    const speed = Units[candidate.piece.type].mobility;
    let options = 0;
    for (
      let row = Math.max(0, candidate.at[0] - speed);
      row <= Math.min(7, candidate.at[0] + speed);
      row++
    ) {
      for (
        let column = Math.max(0, candidate.at[1] - speed);
        column <= Math.min(7, candidate.at[1] + speed);
        column++
      ) {
        if (
          (row !== candidate.at[0] || column !== candidate.at[1]) &&
          chebyshev(candidate.at, [row, column]) <= speed &&
          board[row][column] === null
        ) {
          options++;
        }
      }
    }
    relocationOptions += options;
    if (options === 0) {
      immobileCount++;
      if (candidate.at[0] === ownHomeRank) homeRankImmobileCount++;
    }
  }

  const remaining = new Set(material.map(({ at }) => at.join(",")));
  const byKey = new Map(material.map((piece) => [piece.at.join(","), piece]));
  const componentSizes: number[] = [];
  while (remaining.size > 0) {
    const first = remaining.values().next().value as string;
    remaining.delete(first);
    const stack = [byKey.get(first)!];
    let size = 0;
    while (stack.length > 0) {
      const current = stack.pop()!;
      size++;
      for (const key of [...remaining]) {
        const other = byKey.get(key)!;
        if (chebyshev(current.at, other.at) <= 1) {
          remaining.delete(key);
          stack.push(other);
        }
      }
    }
    componentSizes.push(size);
  }
  return {
    relocationOptions,
    meanRelocationOptions:
      material.length === 0 ? 0 : relocationOptions / material.length,
    immobileCount,
    homeRankImmobileCount,
    connectedComponents: componentSizes.length,
    largestComponentRatio:
      material.length === 0 ? 1 : Math.max(...componentSizes) / material.length,
  };
}

function extractStructureV2SideFeatures(
  position: ValuePosition,
  player: Player
): StructureV2SideFeatures {
  const friendly = piecesFor(position.board, player);
  const ownHomeRank = homeRank(player);
  const formationInfantry = friendly.filter(
    ({ piece }) =>
      INFANTRY_TYPES.has(piece.type) && piece.type !== "AIRBORNE_INFANTRY"
  );
  const formationMaterial = friendly.filter(
    ({ piece }) => piece.type !== "HQ" && piece.type !== "AIRBORNE_INFANTRY"
  );
  const result = Object.fromEntries(
    STRUCTURE_V2_SIDE_FEATURE_NAMES.map((name) => [name, 0])
  ) as StructureV2SideFeatures;

  // A staggered infantry screen has diagonal adjacency without forming an
  // easily penetrated same-file column. These raw counts let the model learn
  // the phase-dependent trade-off instead of baking a fixed score into search.
  for (let first = 0; first < formationInfantry.length; first++) {
    for (let second = first + 1; second < formationInfantry.length; second++) {
      const rowDistance = Math.abs(
        formationInfantry[first].at[0] - formationInfantry[second].at[0]
      );
      const columnDistance = Math.abs(
        formationInfantry[first].at[1] - formationInfantry[second].at[1]
      );
      if (rowDistance === 1 && columnDistance === 0) {
        result.infantry_vertical_adjacent_pairs++;
      }
      if (rowDistance === 1 && columnDistance === 1) {
        result.infantry_diagonal_adjacent_pairs++;
      }
    }
  }
  for (let file = 0; file < 8; file++) {
    const rows = formationInfantry
      .filter(({ at }) => at[1] === file)
      .map(({ at }) => at[0])
      .sort((left, right) => left - right);
    let run = 1;
    for (let index = 1; index < rows.length; index++) {
      if (rows[index] === rows[index - 1] + 1) {
        run++;
        if (run >= 3) result.infantry_same_file_run_excess++;
      } else {
        run = 1;
      }
    }
  }
  result.infantry_isolated_count = formationInfantry.filter((candidate) =>
    formationMaterial.every(
      (other) => other === candidate || chebyshev(candidate.at, other.at) > 1
    )
  ).length;

  const infantryFiles = formationInfantry.map(({ at }) => at[1]);
  const infantryRanks = formationInfantry.map(({ at }) => at[0]);
  result.infantry_distinct_files = new Set(infantryFiles).size;
  result.infantry_file_span = infantryFiles.length
    ? Math.max(...infantryFiles) - Math.min(...infantryFiles)
    : 0;
  result.infantry_rank_span = infantryRanks.length
    ? Math.max(...infantryRanks) - Math.min(...infantryRanks)
    : 0;
  if (formationInfantry.length) {
    const frontierAdvance = Math.max(
      ...formationInfantry.map(({ at }) => Math.abs(at[0] - ownHomeRank))
    );
    result.infantry_frontier_count = formationInfantry.filter(
      ({ at }) => Math.abs(at[0] - ownHomeRank) === frontierAdvance
    ).length;
  }

  const materialFiles = formationMaterial.map(({ at }) => at[1]);
  const materialRanks = formationMaterial.map(({ at }) => at[0]);
  result.material_file_span = materialFiles.length
    ? Math.max(...materialFiles) - Math.min(...materialFiles)
    : 0;
  result.material_rank_span = materialRanks.length
    ? Math.max(...materialRanks) - Math.min(...materialRanks)
    : 0;
  let materialPairDistance = 0;
  let materialPairs = 0;
  for (let first = 0; first < formationMaterial.length; first++) {
    for (let second = first + 1; second < formationMaterial.length; second++) {
      materialPairDistance += chebyshev(
        formationMaterial[first].at,
        formationMaterial[second].at
      );
      materialPairs++;
    }
  }
  result.material_pair_distance_mean = materialPairs
    ? materialPairDistance / materialPairs
    : 0;
  return result;
}

function extractTacticalV3SideFeatures(
  position: ValuePosition,
  player: Player
): TacticalV3SideFeatures {
  const friendly = piecesFor(position.board, player);
  const enemy = piecesFor(position.board, otherPlayer(player));
  const result = Object.fromEntries(
    TACTICAL_V3_SIDE_FEATURE_NAMES.map((name) => [name, 0])
  ) as TacticalV3SideFeatures;
  const hq = friendly.find(({ piece }) => piece.type === "HQ");
  if (!hq) return result;

  const enemyInfantry = enemy.filter(({ piece }) =>
    INFANTRY_TYPES.has(piece.type)
  );
  const friendlyInfantry = friendly.filter(({ piece }) =>
    INFANTRY_TYPES.has(piece.type)
  );
  const distance = (candidate: PlacedPiece) => manhattan(hq.at, candidate.at);
  const minimumDistance = (type: UnitType) => {
    const distances = enemyInfantry
      .filter(({ piece }) => piece.type === type)
      .map(distance);
    return distances.length ? Math.min(...distances) : 15;
  };

  result.hq_enemy_infantry_distance_min = enemyInfantry.length
    ? Math.min(...enemyInfantry.map(distance))
    : 15;
  result.hq_enemy_armored_infantry_distance_min =
    minimumDistance("ARMORED_INFANTRY");
  result.hq_enemy_airborne_infantry_distance_min =
    minimumDistance("AIRBORNE_INFANTRY");
  result.hq_enemy_infantry_within_two = enemyInfantry.filter(
    (candidate) => distance(candidate) <= 2
  ).length;
  result.hq_enemy_infantry_within_three = enemyInfantry.filter(
    (candidate) => distance(candidate) <= 3
  ).length;
  result.hq_friendly_infantry_within_two = friendlyInfantry.filter(
    (candidate) => distance(candidate) <= 2
  ).length;
  result.hq_friendly_infantry_within_three = friendlyInfantry.filter(
    (candidate) => distance(candidate) <= 3
  ).length;
  for (const attacker of enemyInfantry) {
    const proximity = Math.max(0, 4 - distance(attacker));
    const weight =
      attacker.piece.type === "ARMORED_INFANTRY"
        ? 1.5
        : attacker.piece.type === "AIRBORNE_INFANTRY"
        ? 1.25
        : 1;
    result.hq_attack_pressure += proximity * weight;
  }
  for (const defender of friendlyInfantry) {
    result.hq_defense_density += Math.max(0, 4 - distance(defender));
  }
  return result;
}

function extractSideFeatures(
  position: ValuePosition,
  player: Player
): SideFeatures {
  const board = position.board;
  const opponent = otherPlayer(player);
  const friendly = piecesFor(board, player);
  const enemy = piecesFor(board, opponent);
  const reserve = reserveFor(position, player);
  const artillery = friendly.filter(({ piece }) =>
    ARTILLERY_TYPES.has(piece.type)
  );
  const infantry = friendly.filter(({ piece }) =>
    INFANTRY_TYPES.has(piece.type)
  );
  const bombardment = bombardedSquares(board);
  const ownHomeRank = homeRank(player);
  const counts = Object.fromEntries(
    PIECE_TYPES.map((type) => [type, 0])
  ) as Record<(typeof PIECE_TYPES)[number], number>;
  friendly.forEach(
    ({ piece }) => counts[piece.type as (typeof PIECE_TYPES)[number]]++
  );

  const result = Object.fromEntries(
    SIDE_FEATURE_NAMES.map((name) => [name, 0])
  ) as SideFeatures;
  PIECE_TYPES.forEach((type) => {
    result[`board_${type.toLowerCase()}` as SideFeatureName] = counts[type];
  });
  RESERVE_TYPES.forEach((type) => {
    result[`reserve_${type.toLowerCase()}` as SideFeatureName] =
      reserve[type] ?? 0;
  });

  result.material_board = friendly.reduce(
    (total, { piece }) => total + unitScores[piece.type],
    0
  );
  result.material_total =
    result.material_board +
    RESERVE_TYPES.reduce(
      (total, type) => total + (reserve[type] ?? 0) * unitScores[type],
      0
    );
  result.pieces_board = friendly.length;
  result.pieces_reserve = RESERVE_TYPES.reduce(
    (total, type) => total + (reserve[type] ?? 0),
    0
  );
  result.infantry_board = infantry.length;
  result.artillery_board = artillery.length;
  result.home_rank_occupancy = friendly.filter(
    ({ at }) => at[0] === ownHomeRank
  ).length;
  const structure = structureAndOptionMetrics(board, friendly, ownHomeRank);
  result.relocation_options = structure.relocationOptions;
  result.mean_relocation_options = structure.meanRelocationOptions;
  result.immobile_count = structure.immobileCount;
  result.home_rank_immobile_count = structure.homeRankImmobileCount;
  result.connected_components = structure.connectedComponents;
  result.largest_component_ratio = structure.largestComponentRatio;

  const rankCounts = Array(8).fill(0) as number[];
  friendly.forEach(({ at }) => rankCounts[at[0]]++);
  result.max_rank_occupancy = Math.max(0, ...rankCounts);
  const advances = friendly
    .filter(({ piece }) => piece.type !== "HQ")
    .map(({ at }) => Math.abs(at[0] - ownHomeRank));
  result.advancement_mean =
    advances.length === 0
      ? 0
      : advances.reduce((total, value) => total + value, 0) / advances.length;
  result.advancement_max = Math.max(0, ...advances);

  const eligibleForSupport = friendly.filter(
    ({ piece }) => piece.type !== "HQ" && piece.type !== "AIRBORNE_INFANTRY"
  );
  const rankPower = Array(8).fill(0) as number[];
  eligibleForSupport.forEach(({ at, piece }) => {
    rankPower[at[0]] += unitScores[piece.type];
  });
  const anchorRank = rankPower.indexOf(Math.max(...rankPower));
  let supportDistanceTotal = 0;
  for (const candidate of eligibleForSupport) {
    const distance = nearestSupportDistance(candidate, friendly);
    supportDistanceTotal += distance;
    if (distance > 1) {
      result.unsupported_count++;
      result.unsupported_value += unitScores[candidate.piece.type];
    }
    const ranksPastAnchor =
      player === "RED"
        ? anchorRank - candidate.at[0]
        : candidate.at[0] - anchorRank;
    if (ranksPastAnchor >= 2) {
      result.overextended_count++;
      result.overextended_value +=
        unitScores[candidate.piece.type] * (ranksPastAnchor - 1);
    }
  }
  result.support_distance_mean =
    eligibleForSupport.length === 0
      ? 0
      : supportDistanceTotal / eligibleForSupport.length;

  for (let first = 0; first < artillery.length; first++) {
    for (let second = first + 1; second < artillery.length; second++) {
      if (chebyshev(artillery[first].at, artillery[second].at) === 1) {
        result.artillery_adjacent_pairs++;
      }
    }
  }
  result.heavy_artillery_centered = artillery.some(
    (gun) =>
      gun.piece.type === "HEAVY_ARTILLERY" && isHeavyCentered(gun, artillery)
  )
    ? 1
    : 0;

  for (const gun of artillery) {
    const orientation = gun.piece.orientation ?? (player === "RED" ? 0 : 180);
    if (orientation % 90 === 0) result.artillery_cardinal_count++;
    else result.artillery_diagonal_count++;
    let protectedGun = false;
    for (const at of neighbors(gun.at, true)) {
      const piece = board[at[0]][at[1]];
      if (piece?.player !== player || !INFANTRY_TYPES.has(piece.type)) continue;
      protectedGun = true;
      if (at[0] !== gun.at[0] && at[1] !== gun.at[1]) {
        result.artillery_diagonal_infantry_cover++;
      } else {
        result.artillery_cardinal_infantry_cover++;
      }
    }
    if (protectedGun) result.artillery_protected_count++;
  }

  for (const [key, control] of Object.entries(bombardment)) {
    if (!control[player]) continue;
    result.bombarded_squares++;
    const [row, column] = key.split(",").map(Number);
    const target = board[row][column];
    if (target?.player === opponent) {
      result.bombarded_enemy_count++;
      result.bombarded_enemy_value += unitScores[target.type];
    }
  }

  const seenEngagements = new Set<string>();
  for (const unit of infantry) {
    for (const at of neighbors(unit.at, false)) {
      const target = board[at[0]][at[1]];
      if (target?.player === opponent && INFANTRY_TYPES.has(target.type)) {
        const key = [unit.at.join(","), at.join(",")].sort().join("|");
        seenEngagements.add(key);
      }
    }
  }
  result.infantry_engagements = seenEngagements.size;

  const paratroopers = friendly.filter(
    ({ piece }) => piece.type === "AIRBORNE_INFANTRY"
  );
  result.paratrooper_ready =
    (reserve.AIRBORNE_INFANTRY ?? 0) > 0 ||
    paratroopers.some(({ at }) => at[0] === ownHomeRank)
      ? 1
      : 0;
  const deployedParatroopers = paratroopers.filter(
    ({ at }) => at[0] !== ownHomeRank
  );
  result.paratrooper_deployed = deployedParatroopers.length;
  for (const para of deployedParatroopers) {
    result.paratrooper_distance_home += Math.abs(para.at[0] - ownHomeRank);
    if (
      neighbors(para.at, true).some(([row, column]) => {
        const piece = board[row][column];
        return piece?.player === player && piece.type !== "HQ";
      })
    ) {
      result.paratrooper_supported++;
    }
    if (
      neighbors(para.at, false).some(([row, column]) => {
        const piece = board[row][column];
        return piece?.player === opponent && INFANTRY_TYPES.has(piece.type);
      })
    ) {
      result.paratrooper_engaged++;
    }
  }

  const hq = friendly.find(({ piece }) => piece.type === "HQ");
  if (hq) {
    result.hq_bombarded = bombardment[hq.at.join(",")]?.[opponent] ? 1 : 0;
    for (const at of neighbors(hq.at, true)) {
      const piece = board[at[0]][at[1]];
      if (piece?.player === player) result.hq_adjacent_friendly++;
      if (
        piece?.player === opponent &&
        INFANTRY_TYPES.has(piece.type) &&
        (at[0] === hq.at[0] || at[1] === hq.at[1])
      ) {
        result.hq_adjacent_enemy_infantry++;
      }
      if (piece === null && !bombardment[at.join(",")]?.[opponent]) {
        const enemyInfantryAdjacent = neighbors(at, false).some(
          ([row, column]) => {
            const neighbor = board[row][column];
            return (
              neighbor?.player === opponent && INFANTRY_TYPES.has(neighbor.type)
            );
          }
        );
        if (!enemyInfantryAdjacent) result.hq_escape_squares++;
      }
    }
  } else {
    result.hq_bombarded = 1;
  }
  result.pseudo_mobility = pseudoMobility(board, friendly);
  return result;
}

export function extractValueFeatures(
  position: ValuePosition,
  perspective: Player
): number[] {
  return extractValueFeaturesWithSchema(
    position,
    perspective,
    SIDE_FEATURE_NAMES,
    DIFFERENCE_FEATURES,
    VALUE_FEATURE_NAMES.length
  );
}

export function extractValueFeaturesV2(
  position: ValuePosition,
  perspective: Player
): number[] {
  const opponent = otherPlayer(perspective);
  const own = extractStructureV2SideFeatures(position, perspective);
  const opp = extractStructureV2SideFeatures(position, opponent);
  const values = [
    ...extractValueFeatures(position, perspective),
    ...STRUCTURE_V2_SIDE_FEATURE_NAMES.map((name) => own[name]),
    ...STRUCTURE_V2_SIDE_FEATURE_NAMES.map((name) => opp[name]),
    ...STRUCTURE_V2_SIDE_FEATURE_NAMES.map((name) => own[name] - opp[name]),
  ];
  if (
    values.length !== VALUE_FEATURE_NAMES_V2.length ||
    values.some((value) => !Number.isFinite(value))
  ) {
    throw new Error("Invalid GHQ v2 value feature vector");
  }
  return values;
}

export function extractValueFeaturesV3(
  position: ValuePosition,
  perspective: Player
): number[] {
  const opponent = otherPlayer(perspective);
  const own = extractTacticalV3SideFeatures(position, perspective);
  const opp = extractTacticalV3SideFeatures(position, opponent);
  const values = [
    ...extractValueFeaturesV2(position, perspective),
    ...TACTICAL_V3_SIDE_FEATURE_NAMES.map((name) => own[name]),
    ...TACTICAL_V3_SIDE_FEATURE_NAMES.map((name) => opp[name]),
    ...TACTICAL_V3_SIDE_FEATURE_NAMES.map((name) => own[name] - opp[name]),
  ];
  if (
    values.length !== VALUE_FEATURE_NAMES_V3.length ||
    values.some((value) => !Number.isFinite(value))
  ) {
    throw new Error("Invalid GHQ v3 value feature vector");
  }
  return values;
}

function extractValueFeaturesWithSchema(
  position: ValuePosition,
  perspective: Player,
  sideFeatureNames: readonly SideFeatureName[],
  differenceFeatures: readonly SideFeatureName[],
  expectedLength: number
): number[] {
  const opponent = otherPlayer(perspective);
  const own = extractSideFeatures(position, perspective);
  const opp = extractSideFeatures(position, opponent);
  const totalReserve = PLAYERS.reduce(
    (total, player) =>
      total +
      RESERVE_TYPES.reduce(
        (subtotal, type) => subtotal + reserveFor(position, player)[type],
        0
      ),
    0
  );
  const occupied = position.board.flat().filter(Boolean).length;
  const nonHqOnBoard = position.board
    .flat()
    .filter((piece) => piece !== null && piece.type !== "HQ").length;
  const values = [
    Math.min(Math.max(position.turnNumber, 0), 100) / 100,
    occupied / 64,
    // Standard GHQ begins with 13 non-HQ units per player.
    (nonHqOnBoard + totalReserve) / 26,
    position.currentPlayer === perspective ? 1 : 0,
    ...sideFeatureNames.map((name) => own[name]),
    ...sideFeatureNames.map((name) => opp[name]),
    ...differenceFeatures.map((name) => own[name] - opp[name]),
  ];
  if (
    values.length !== expectedLength ||
    values.some((value) => !Number.isFinite(value))
  ) {
    throw new Error("Invalid GHQ value feature vector");
  }
  return values;
}

export function valueFeaturesToRecord(features: number[]): ValueFeatureRecord {
  if (features.length !== VALUE_FEATURE_NAMES.length) {
    throw new Error(
      `Expected ${VALUE_FEATURE_NAMES.length} value features, received ${features.length}`
    );
  }
  return Object.fromEntries(
    VALUE_FEATURE_NAMES.map((name, index) => [name, features[index]])
  ) as ValueFeatureRecord;
}

export function extractValueFeatureRecord(
  position: ValuePosition,
  perspective: Player
): ValueFeatureRecord {
  return valueFeaturesToRecord(extractValueFeatures(position, perspective));
}
