import { GHQState } from "@/game/engine";

export const initialBoardSetup: GHQState["board"] = [
  [
    { type: "HQ", player: "BLUE" },
    { type: "ARTILLERY", player: "BLUE", orientation: 180 },
    null,
    null,
    null,
    null,
    null,
    null,
  ],
  [
    { type: "INFANTRY", player: "BLUE" },
    { type: "INFANTRY", player: "BLUE" },
    { type: "INFANTRY", player: "BLUE" },
    null,
    null,
    null,
    null,
    null,
  ],
  [null, null, null, null, null, null, null, null],
  [null, null, null, null, null, null, null, null],
  [null, null, null, null, null, null, null, null],
  [null, null, null, null, null, null, null, null],
  [
    null,
    null,
    null,
    null,
    null,
    { type: "INFANTRY", player: "RED" },
    { type: "INFANTRY", player: "RED" },
    { type: "INFANTRY", player: "RED" },
  ],
  [
    null,
    null,
    null,
    null,
    null,
    null,
    { type: "ARTILLERY", player: "RED", orientation: 0 },
    { type: "HQ", player: "RED" },
  ],
];

export const initialBoardSetupWithAnArmored: GHQState["board"] = [
  [
    { type: "HQ", player: "BLUE" },
    { type: "ARTILLERY", player: "BLUE", orientation: 180 },
    null,
    null,
    null,
    null,
    null,
    { type: "ARMORED_INFANTRY", player: "BLUE", orientation: 180 },
  ],
  [
    { type: "INFANTRY", player: "BLUE" },
    { type: "INFANTRY", player: "BLUE" },
    { type: "INFANTRY", player: "BLUE" },
    null,
    null,
    null,
    null,
    null,
  ],
  [null, null, null, null, null, null, null, null],
  [null, null, null, null, null, null, null, null],
  [null, null, null, null, null, null, null, null],
  [null, null, null, null, null, null, null, null],
  [
    null,
    null,
    null,
    null,
    null,
    { type: "INFANTRY", player: "RED" },
    { type: "INFANTRY", player: "RED" },
    { type: "INFANTRY", player: "RED" },
  ],
  [
    null,
    null,
    null,
    null,
    null,
    null,
    { type: "ARTILLERY", player: "RED", orientation: 0 },
    { type: "HQ", player: "RED" },
  ],
];

export const initialBoardSetupWithAnAirborneBack: GHQState["board"] = [
  [
    { type: "HQ", player: "BLUE" },
    { type: "ARTILLERY", player: "BLUE", orientation: 180 },
    null,
    null,
    null,
    null,
    null,
    { type: "AIRBORNE_INFANTRY", player: "BLUE" },
  ],
  [
    { type: "INFANTRY", player: "BLUE" },
    { type: "INFANTRY", player: "BLUE" },
    { type: "INFANTRY", player: "BLUE" },
    null,
    null,
    null,
    null,
    null,
  ],
  [null, null, null, null, null, null, null, null],
  [null, null, null, null, null, null, null, null],
  [null, null, null, null, null, null, null, null],
  [null, null, null, null, null, null, null, null],
  [
    null,
    null,
    null,
    null,
    null,
    { type: "INFANTRY", player: "RED" },
    { type: "INFANTRY", player: "RED" },
    { type: "INFANTRY", player: "RED" },
  ],
  [
    null,
    null,
    null,
    null,
    null,
    null,
    { type: "ARTILLERY", player: "RED", orientation: 0 },
    { type: "HQ", player: "RED" },
  ],
];

export const initialBoardSetupWithAnAirborneNotBack: GHQState["board"] = [
  [
    { type: "HQ", player: "BLUE" },
    { type: "ARTILLERY", player: "BLUE", orientation: 180 },
    null,
    null,
    null,
    null,
    null,
    null,
  ],
  [
    { type: "INFANTRY", player: "BLUE" },
    { type: "INFANTRY", player: "BLUE" },
    { type: "INFANTRY", player: "BLUE" },
    { type: "AIRBORNE_INFANTRY", player: "BLUE" },
    null,
    null,
    null,
    null,
  ],
  [null, null, null, null, null, null, null, null],
  [null, null, null, null, null, null, null, null],
  [null, null, null, null, null, null, null, null],
  [null, null, null, null, null, null, null, null],
  [
    null,
    null,
    null,
    null,
    null,
    { type: "INFANTRY", player: "RED" },
    { type: "INFANTRY", player: "RED" },
    { type: "INFANTRY", player: "RED" },
  ],
  [
    null,
    null,
    null,
    null,
    null,
    null,
    { type: "ARTILLERY", player: "RED", orientation: 0 },
    { type: "HQ", player: "RED" },
  ],
];

export const artillaryFaceOff: GHQState["board"] = [
  [null, null, null, null, null, null, null, null],
  [null, null, null, null, null, null, null, null],
  [null, null, null, null, null, null, null, null],
  [
    null,
    { type: "ARTILLERY", player: "RED", orientation: 90 },
    null,
    { type: "ARTILLERY", player: "BLUE", orientation: 270 },
    null,
    null,
    null,
    null,
  ],
  [null, null, null, null, null, null, null, null],
  [null, null, null, null, null, null, null, null],
  [null, null, null, null, null, null, null, null],
  [null, null, null, null, null, null, null, null],
];