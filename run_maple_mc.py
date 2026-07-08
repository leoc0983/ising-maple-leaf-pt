import numpy as np
import argparse
from pathlib import Path
from numba import njit

def maple_leaf_lattice_pbc(nx, ny):
    t1 = np.array([1.0, 0.0])
    t2 = np.array([0.5, np.sqrt(3) / 2])

    R = np.array([[3, 1],
                  [-1, 2]])

    Rinv = np.linalg.inv(R)
    sites = []
    coords = []

    for m in range(nx):
        for n in range(ny):
            pq = Rinv @ np.array([m, n])
            if np.all(np.abs(pq - np.round(pq)) < 1e-6):
                continue

            sites.append(m * t1 + n * t2)
            coords.append((m, n))

    sites = np.array(sites)
    coord_to_site = {c: i for i, c in enumerate(coords)}
    nn_dirs = [(1, 0), (0, 1), (1, -1),]

    bonds = set()
    for i, (m, n) in enumerate(coords):
        for dm, dn in nn_dirs:
            m2 = (m + dm) % nx
            n2 = (n + dn) % ny

            if (m2, n2) not in coord_to_site: continue

            j = coord_to_site[(m2, n2)]
            bonds.add(tuple(sorted((i, j))))
    return sites, coords, coord_to_site, sorted(bonds)

def build_neighbors(N, bonds):
    nbrs = [[] for _ in range(N)]
    for i, j in bonds:
        nbrs[i].append(j)
        nbrs[j].append(i)
    return nbrs

@njit(cache=True, fastmath=True)
def total_energy(spins, bonds, J):
    E = 0.0
    for k in range(bonds.shape[0]):
        i = bonds[k,0]
        j = bonds[k,1]
        E -= J * spins[i] * spins[j]
    return E

@njit(cache=True, fastmath=True)
def delta_energy(i, spins, neighbors, J):
    s = 0
    for k in range(neighbors.shape[1]):
        s += spins[neighbors[i, k]]
    return 2.0 * J * spins[i] * s

@njit(cache=True, fastmath=True)
def mc_sweep(spins, neighbors, T, J, E): #Monte Carlo step
    N = len(spins)
    for _ in range(N):
        i = np.random.randint(N)
        dE = delta_energy(i, spins, neighbors, J)
        if dE <= 0.0:
            spins[i] *= -1
            E += dE
        else:
            if np.random.random() < np.exp(-dE / T):
                spins[i] *= -1
                E += dE
    return E

def attempt_swap(spins_list, energies, temps):
    accepted = 0
    for r in range(len(temps) - 1):
        beta1 = 1.0 / temps[r]
        beta2 = 1.0 / temps[r + 1]
        delta = (beta1 - beta2) * (energies[r] - energies[r + 1])
        if delta >= 0 or np.random.rand() < np.exp(delta):
            spins_list[r], spins_list[r + 1] = \
                spins_list[r + 1], spins_list[r]
            energies[r], energies[r + 1] = \
                energies[r + 1], energies[r]
            accepted += 1
    return accepted

def run_pt(bonds, neighbors, temps, J, n_therm, n_meas, sweeps_per_exchange):
    N = len(neighbors)
    nbonds = len(bonds)
    nrep = len(temps)
    spins_list = [np.where(np.random.random(N) < 0.5, -1, 1).astype(np.int8) for _ in range(nrep)]
    energies = [total_energy(s, bonds, J) for s in spins_list]

    # ---------- thermalization ----------
    for step in range(n_therm):
        for r in range(nrep):
            for _ in range(sweeps_per_exchange):
                energies[r] = mc_sweep(spins_list[r], neighbors, temps[r], J, energies[r])

        attempt_swap(spins_list, energies, temps)
        if step % 500 == 0:
            print(f"thermalization {step}/{n_therm}, E/bond={energies[0]/nbonds:.6f}", flush=True)

    # ---------- measurement ----------
    spin_configs = []
    E_history = []
    #corr_x_acc = np.zeros(len(pairs_x))
    #corr_y_acc = np.zeros(len(pairs_y))

    for step in range(n_meas):
        for r in range(nrep):
            for _ in range(sweeps_per_exchange):
                energies[r] = mc_sweep(spins_list[r], neighbors, temps[r], J, energies[r])
        
        spin_configs.append(spins_list[0].copy())
        E_history.append(energies[0])
        attempt_swap(spins_list, energies, temps)
        
        if step % 1000 == 0:
            print(f"measurement {step}/{n_meas}, E/bond={energies[0]/nbonds:.6f}", flush=True)

    return {
        "bonds": bonds,
        "temps": temps,
        "spin_configs": np.array(spin_configs, dtype=np.int8),
        "E_history": np.array(E_history),
        "n_therm": n_therm,
        "n_meas": n_meas,
        "sweeps_per_exchange": sweeps_per_exchange,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, required=True)
    parser.add_argument("--Tmin", type=float, default=0.02)
    parser.add_argument("--Tmax", type=float, default=2.0)
    parser.add_argument("--nrep", type=int, default=48)
    parser.add_argument("--ntherm", type=int, default=5000)
    parser.add_argument("--nmeas", type=int, default=10000)
    parser.add_argument("--sweeps", type=int, default=20)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    assert args.L % 7 == 0

    sites, coords, coord_to_site, bonds = maple_leaf_lattice_pbc(args.L, args.L)
    bonds = np.array(bonds, dtype=np.int32)

    neighbors = build_neighbors(len(sites), bonds)
    neighbors = np.array(neighbors, dtype=np.int32)

    temps = np.geomspace(args.Tmin, args.Tmax, args.nrep)

    result = run_pt(bonds, neighbors, temps, J=-1.0, n_therm=args.ntherm, n_meas=args.nmeas, sweeps_per_exchange=args.sweeps)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, L=args.L, N=len(sites), **result)

    print("\n========== DONE ==========")
    print("L =", args.L)
    print("N =", len(sites))

if __name__ == "__main__":
    main()