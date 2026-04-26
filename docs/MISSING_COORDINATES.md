# Missing Coordinates - Hit List

This document tracks coordinates that need to be verified or filled in from actual gameplay testing.

## Format
- ❌ = Missing/unverified coordinate
- ⚠️ = Coordinate exists but needs verification
- ✅ = Confirmed accurate coordinate

---

## LITTLEROOT TOWN
- ✅ North exit to Route 101: `(10, 0)` - exit coords
- ⚠️ Entry from Route 101: `(10, 28)` - needs verification
- ✅ Player's House door: `(4, 7)` - confirmed from gameplay script
- ✅ May's House door: `(14, 8)` - confirmed from gameplay script
- ✅ Birch's Lab door: `(7, 16)` - confirmed from gameplay script

## PLAYER'S HOUSE
- ✅ 1F exit door (inside): `(4, 7)` - confirmed from gameplay script
- ✅ 1F stairs: `(8, 2)` - confirmed from gameplay script
- ✅ 2F stairs: `(8, 2)` - confirmed from gameplay script
- ⚠️ 2F clock interaction: `(5, 1)` - from gameplay script, verify

## MAY'S HOUSE
- ✅ 1F exit door (inside): `(2, 9)` - confirmed from gameplay script
- ✅ 1F stairs: `(2, 2)` - confirmed from gameplay script
- ⚠️ 2F stairs: `(1, 1)` - needs verification (gameplay script says 1,1 for exit)
- ✅ 2F Pokeball interaction: `(5, 4)` - confirmed from gameplay script

## PROFESSOR BIRCH'S LAB
- ✅ Exit door (inside): `(6, 13)` - confirmed from gameplay script and objective manager
- ⚠️ Birch's bag on Route 101: `(7, 14)` - from gameplay script, needs verification

## ROUTE 101
- ⚠️ South exit to Littleroot: `(10, 28)` - needs verification
- ⚠️ Entry from Littleroot: `(10, 0)` - needs verification
- ⚠️ North exit to Oldale: `(11, 0)` - from objective manager, needs verification
- ⚠️ Entry from Oldale: `(11, 24)` - estimated, needs verification

## OLDALE TOWN
- ⚠️ South exit to Route 101: `(10, 19)` - from objective manager, needs verification
- ⚠️ Entry from Route 101: `(10, 0)` - estimated, needs verification
- ⚠️ North exit to Route 103: `(10, 0)` - estimated, needs verification
- ⚠️ Entry from Route 103: `(10, 22)` - from objective manager, needs verification
- ⚠️ West exit to Route 102: `(0, 9)` - estimated, needs verification
- ⚠️ Entry from Route 102: `(25, 9)` - estimated, needs verification
- ✅ Pokemon Center door: `(6, 16)` - confirmed from gameplay script
- ⚠️ Pokemon Center door (inside): `(7, 9)` - estimated, needs verification
- ⚠️ Mart door: `(13, 16)` - estimated, needs verification
- ⚠️ Mart door (inside): `(3, 7)` - estimated, needs verification

## ROUTE 103
- ⚠️ South exit to Oldale: `(10, 22)` - from objective manager, needs verification
- ⚠️ Entry from Oldale: `(10, 0)` - estimated, needs verification
- ✅ Rival May position: `(9, 3)` - confirmed from gameplay script and objective manager

## ROUTE 102
- ⚠️ East exit to Oldale: `(48, 9)` - estimated, needs verification
- ⚠️ Entry from Oldale: `(0, 9)` - estimated, needs verification
- ⚠️ West exit to Petalburg: `(0, 12)` - estimated, needs verification
- ⚠️ Entry from Petalburg: `(48, 12)` - estimated, needs verification
- ✅ Trainer 1 (Youngster Calvin): `(33, 15)` - confirmed from gameplay script
- ✅ Trainer 2 (Bug Catcher Rick): `(25, 14)` - confirmed from gameplay script
- ✅ Trainer 3 (Youngster Allen): `(19, 7)` - confirmed from gameplay script

## PETALBURG CITY
- ⚠️ East exit to Route 102: `(29, 17)` - estimated, needs verification
- ⚠️ Entry from Route 102: `(0, 12)` - estimated, needs verification
- ⚠️ North exit to Route 104 South: `(17, 0)` - estimated, needs verification
- ⚠️ Entry from Route 104 South: `(17, 27)` - estimated, needs verification
- ⚠️ Gym door: `(15, 8)` - from gameplay script, needs verification
- ⚠️ Gym door (inside): `(4, 112)` - from gameplay script, needs verification (unusual Y value)
- ⚠️ Pokemon Center door: `(23, 8)` - estimated (was 6,8 — clearly wrong; gym is at 15,8 so PC must be east of gym, closer to Route 102 entry)
- ⚠️ Pokemon Center door (inside): `(7, 9)` - estimated, needs verification

## PETALBURG CITY GYM
- ⚠️ Norman's position: `(15, 9)` OR `(4, 107)` - conflicting info in gameplay script, needs verification
- ⚠️ Exit door (inside): `(4, 112)` - needs verification

## ROUTE 104 (SOUTH)
- ⚠️ South exit to Petalburg: `(17, 27)` - estimated, needs verification
- ⚠️ Entry from Petalburg: `(17, 0)` - estimated, needs verification
- ⚠️ North exit to Petalburg Woods: `(9, 0)` - estimated, needs verification
- ⚠️ Entry from Petalburg Woods: `(9, 47)` - estimated, needs verification
- ⚠️ Trainer (Rich Boy Winston): `(11, 43)` - from gameplay script, needs verification

## PETALBURG WOODS
- ⚠️ South exit to Route 104 South: `(9, 47)` - estimated, needs verification
- ⚠️ Entry from Route 104 South: `(9, 0)` - estimated, needs verification
- ⚠️ North exit to Route 104 North: `(7, 0)` - estimated, needs verification
- ⚠️ Entry from Route 104 North: `(7, 47)` - estimated, needs verification
- ✅ Trainer 1 (Bug Catcher Lyle): `(9, 32)` - confirmed from gameplay script
- ✅ Team Aqua Grunt: `(26, 23)` - confirmed from gameplay script
- ✅ Trainer 2 (Bug Catcher James): `(7, 14)` - confirmed from gameplay script

## ROUTE 104 (NORTH)
- ⚠️ South exit to Petalburg Woods: `(7, 47)` - estimated, needs verification
- ⚠️ Entry from Petalburg Woods: `(7, 0)` - estimated, needs verification
- ⚠️ North exit to Rustboro: `(19, 0)` - estimated, needs verification
- ⚠️ Entry from Rustboro: `(19, 47)` - estimated, needs verification
- ✅ Trainer 1 (Lass Haley): `(19, 25)` - confirmed from gameplay script
- ⚠️ Trainer 2 (Twins Gina & Mia): `(24, 24)` - from gameplay script, needs verification

## RUSTBORO CITY
- ⚠️ South exit to Route 104 North: `(19, 47)` - estimated, needs verification
- ⚠️ Entry from Route 104 North: `(19, 0)` - estimated, needs verification
- ✅ Pokemon Center door: `(16, 38)` - confirmed from gameplay script
- ⚠️ Pokemon Center door (inside): `(7, 9)` - estimated, needs verification
- ✅ Gym door: `(27, 19)` - confirmed from gameplay script
- ⚠️ Gym door (inside): `(5, 17)` - estimated, needs verification

## RUSTBORO CITY POKEMON CENTER
- ✅ Nurse Joy position: `(7, 3)` - confirmed from gameplay script
- ⚠️ Exit door (inside): `(7, 9)` - estimated, needs verification

## RUSTBORO CITY GYM
- ✅ Trainer (Youngster Josh): `(2, 9)` - confirmed from gameplay script
- ⚠️ Navigate waypoint 1: `(1, 16)` - from gameplay script, needs verification
- ⚠️ Navigate waypoint 2: `(2, 10)` - from gameplay script, needs verification
- ✅ Roxanne position: `(5, 2)` - confirmed from gameplay script
- ⚠️ Exit door (inside): `(5, 17)` - estimated, needs verification

---

## Testing Priority

### HIGH PRIORITY (affects main story progression)
1. Route 101 ↔ Littleroot Town portal coordinates
2. Route 101 ↔ Oldale Town portal coordinates
3. Oldale Town ↔ Route 103 portal coordinates
4. Oldale Town ↔ Route 102 portal coordinates
5. Route 102 ↔ Petalburg City portal coordinates

### MEDIUM PRIORITY (common buildings)
1. Pokemon Center doors (all locations)
2. Gym doors (Petalburg, Rustboro)
3. Birch's Lab coordinates

### LOW PRIORITY (one-time interactions)
1. Player's House internal coordinates
2. May's House internal coordinates
3. Trainer positions (already confirmed from gameplay script)

---

## How to Verify Coordinates

1. **Portal Exit Coords**: Stand on the tile and note position before crossing
2. **Portal Entry Coords**: Note position immediately after crossing
3. **Warp Tiles**: Stand on door/entrance tile and note exact position
4. **Interior Doors**: Note position when inside building near exit

## Coordinate Logging Template

```
Location: [LOCATION_NAME]
Portal Type: [open_world / warp_tile]
From → To: [FROM_LOCATION] → [TO_LOCATION]
Exit Coords (before crossing): (x, y)
Entry Coords (after crossing): (x, y)
Direction: [north/south/east/west/interact]
Notes: [any special observations]
```

---

## Update Procedure

When coordinates are verified:
1. Update this document (change ⚠️ to ✅)
2. Update `agent/location_graph.py` with verified coordinates
3. Run `python agent/location_graph.py` to validate
4. Test pathfinding with updated coordinates
