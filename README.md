# Itokawa Spacecraft Motion Study

Basic setup for studying spacecraft motion around asteroid 25143 Itokawa using a polyhedron gravity model.

---

## Environment Setup

### 1. Create a virtual environment

```bash
python3 -m venv venv
```

### 2. Activate it

```bash
source venv/bin/activate
```

### 3. Upgrade `pip`

```bash
python -m pip install --upgrade pip
```

### 4. Install required packages

```bash
pip install numpy scipy matplotlib polyhedral_gravity pyvista
```

---

## Download the Polyhedral Gravity Source

Clone the ESA polyhedral gravity model repository:

```bash
git clone git@github.com:esa/polyhedral-gravity-model.git
```

---

## Download the Itokawa Shape File

Download the Itokawa 50k poly `.ply` shape file from:

```
https://3d-asteroids.space/asteroids/25143-Itokawa
```

After downloading, rename the file to:

```
itokawa_50k.ply
```

---

## Suggested Folder Layout

```
project_folder/
├── venv/
├── polyhedral-gravity-model/
├── data/
│   └── itokawa_50k.ply
├── scripts/
└── README.md
```

---

## Notes

- Use the 50k poly Itokawa model as the main working shape model.
- Keep the asteroid mesh in the `data/` folder.
- Keep simulation, propagation, and analysis code in the `scripts/` folder.
