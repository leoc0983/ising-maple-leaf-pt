import numpy as np
import argparse
from pathlib import Path

import torch


# ============================================================
# Device
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)



# ============================================================
# Maple Leaf lattice construction
# ============================================================

def maple_leaf_lattice_pbc(nx: int, ny: int):

    t1 = np.array([1.0, 0.0])
    t2 = np.array([0.5, np.sqrt(3) / 2])


    R = np.array([
        [3, 1],
        [-1, 2]
    ])

    Rinv = np.linalg.inv(R)


    sites = []
    coords = []


    for m in range(nx):

        for n in range(ny):

            pq = Rinv @ np.array([m,n])


            if np.all(np.abs(pq - np.round(pq)) < 1e-6):
                continue


            sites.append(
                m*t1 + n*t2
            )

            coords.append(
                (m,n)
            )


    coord_to_site = {
        c:i for i,c in enumerate(coords)
    }



    nn_dirs = [
        (1,0),
        (0,1),
        (1,-1)
    ]


    bonds=set()


    for i,(m,n) in enumerate(coords):

        for dm,dn in nn_dirs:

            m2 = (m+dm)%nx
            n2 = (n+dn)%ny


            if (m2,n2) not in coord_to_site:
                continue


            j = coord_to_site[(m2,n2)]


            if i < j:
                bonds.add((i,j))
            else:
                bonds.add((j,i))


    return (
        np.array(sites),
        coords,
        coord_to_site,
        sorted(bonds)
    )



# ============================================================
# Neighbor list
# ============================================================

def build_neighbors(N, bonds):

    max_deg = 6


    neighbors = -np.ones(
        (N,max_deg),
        dtype=np.int64
    )


    counts = np.zeros(
        N,
        dtype=np.int64
    )


    for i,j in bonds:

        neighbors[i,counts[i]] = j
        counts[i]+=1


        neighbors[j,counts[j]] = i
        counts[j]+=1


    return neighbors



# ============================================================
# Greedy graph coloring
# ============================================================

def color_lattice(N, bonds):

    adjacency=[set() for _ in range(N)]


    for i,j in bonds:

        adjacency[i].add(j)
        adjacency[j].add(i)



    colors=np.full(
        N,
        -1,
        dtype=np.int64
    )


    # order high degree first
    order=sorted(
        range(N),
        key=lambda x:len(adjacency[x]),
        reverse=True
    )


    for node in order:

        used=set()

        for nb in adjacency[node]:

            if colors[nb] >= 0:
                used.add(colors[nb])


        c=0

        while c in used:
            c+=1


        colors[node]=c



    ncolors = colors.max()+1


    color_sites=[]


    for c in range(ncolors):

        color_sites.append(
            np.where(colors==c)[0]
        )


    print(
        f"Using {ncolors} colors"
    )


    for i,c in enumerate(color_sites):

        print(
            f"color {i}: {len(c)} sites"
        )


    return color_sites

# ============================================================
# GPU energy calculation
# ============================================================

def total_energy_gpu(spins, bonds, J):

    # spins:
    # (nrep, N)

    i = bonds[:,0]
    j = bonds[:,1]


    interaction = (
        spins[:,i] *
        spins[:,j]
    )


    return (
        -J *
        interaction.sum(dim=1)
    )



# ============================================================
# GPU color Metropolis update
# ============================================================

@torch.no_grad()
def metropolis_color_update(
    spins,
    neighbors,
    color_sites,
    temps,
    J
):

    """
    Update all sites belonging to one color.

    spins:
        (nrep, N)

    neighbors:
        (N, max_degree)

    color_sites:
        indices of independent spins
    """


    # select spins of this color

    sites = torch.tensor(
        color_sites,
        device=device,
        dtype=torch.long
    )


    # shape:
    # (nrep, number_of_sites_in_color)

    current = spins[:, sites]


    # get neighbors

    nbrs = neighbors[sites]


    # replace invalid neighbors

    valid = nbrs >= 0


    nbrs_safe = nbrs.clone()

    nbrs_safe[~valid] = 0


    nbrs_safe = nbrs_safe.clone().detach().to(
        device=device,
        dtype=torch.long
    )


    valid = valid.clone().detach().to(
        device=device
    )


    # gather neighboring spins
    #
    # result:
    # (nrep, color_sites, neighbors)

    neighbor_spins = spins[:, nbrs_safe]


    neighbor_sum = (
        neighbor_spins *
        valid
    ).sum(dim=2)



    dE = (
        2.0 *
        J *
        current *
        neighbor_sum
    )


    # metropolis probability

    temp = temps[:,None]


    accept = (
        dE <= 0
    )


    random = torch.rand(
        dE.shape,
        device=device
    )


    accept |= (
        random <
        torch.exp(-dE/temp)
    )



    # update

    new_values = current.clone()

    new_values[accept] *= -1


    spins[:, sites] = new_values



# ============================================================
# One full GPU Monte Carlo sweep
# ============================================================

@torch.no_grad()
def mc_sweep_gpu(
    spins,
    neighbors,
    color_sites,
    temps,
    J
):


    for sites in color_sites:

        metropolis_color_update(
            spins,
            neighbors,
            sites,
            temps,
            J
        )



# ============================================================
# Parallel tempering swap
# ============================================================

def attempt_swap(
    spins,
    energies,
    temps
):

    accepted = 0


    for r in range(len(temps)-1):

        beta1 = 1.0 / temps[r]
        beta2 = 1.0 / temps[r+1]


        # move scalar energies to CPU
        E1 = energies[r].item()
        E2 = energies[r+1].item()


        delta = (
            (beta1-beta2)
            *
            (E1-E2)
        )


        if (
            delta >= 0
            or
            np.random.random() < np.exp(delta)
        ):

            # swap GPU tensors directly

            tmp = spins[r].clone()

            spins[r] = spins[r+1]
            spins[r+1] = tmp


            e = energies[r].clone()

            energies[r] = energies[r+1]
            energies[r+1] = e


            accepted += 1


    return accepted

# ============================================================
# Parallel tempering GPU simulation
# ============================================================

def run_pt_gpu(
    bonds,
    neighbors,
    color_sites,
    temps,
    J,
    n_therm,
    n_meas,
    sweeps_per_exchange
):

    nrep = len(temps)
    N = neighbors.shape[0]


    # -----------------------------
    # initialize spins on GPU
    # -----------------------------

    spins = torch.where(
        torch.rand(
            (nrep,N),
            device=device
        ) < 0.5,

        torch.tensor(
            -1,
            device=device,
            dtype=torch.int8
        ),

        torch.tensor(
            1,
            device=device,
            dtype=torch.int8
        )
    )


    temps_gpu = torch.tensor(
        temps,
        device=device,
        dtype=torch.float32
    )


    neighbors_gpu = torch.tensor(
        neighbors,
        device=device,
        dtype=torch.long
    )


    bonds_gpu = torch.tensor(
        bonds,
        device=device,
        dtype=torch.long
    )



    # initial energies

    energies = total_energy_gpu(
        spins,
        bonds_gpu,
        J
    )


    # ========================================================
    # thermalization
    # ========================================================

    for step in range(n_therm):


        for _ in range(sweeps_per_exchange):

            mc_sweep_gpu(
                spins,
                neighbors_gpu,
                color_sites,
                temps_gpu,
                J
            )


        energies = total_energy_gpu(
            spins,
            bonds_gpu,
            J
        )


        if step % 500 == 0:

            print(
                f"thermalization {step}/{n_therm}, "
                f"E/bond={energies[0].item()/len(bonds):.6f}",
                flush=True
            )


        # swap temperatures

        if step % 1 == 0:

            attempt_swap(
                spins,
                energies,
                temps
            )



    # ========================================================
    # measurement
    # ========================================================

    spin_configs=[]
    E_history=[]


    for step in range(n_meas):


        for _ in range(sweeps_per_exchange):

            mc_sweep_gpu(
                spins,
                neighbors_gpu,
                color_sites,
                temps_gpu,
                J
            )


        energies = total_energy_gpu(
            spins,
            bonds_gpu,
            J
        )


        spin_configs.append(
            spins[0]
            .detach()
            .cpu()
            .numpy()
            .copy()
        )


        E_history.append(
            energies[0]
            .item()
        )



        attempt_swap(
            spins,
            energies,
            temps
        )



        if step % 1000 == 0:

            print(
                f"measurement {step}/{n_meas}, "
                f"E/bond={energies[0].item()/len(bonds):.6f}",
                flush=True
            )



    return {

        "spin_configs":
            np.array(
                spin_configs,
                dtype=np.int8
            ),

        "E_history":
            np.array(
                E_history
            ),

        "temps":
            np.array(
                temps
            )

    }



# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--L",
        type=int,
        required=True
    )

    parser.add_argument(
        "--Tmin",
        type=float,
        default=0.02
    )

    parser.add_argument(
        "--Tmax",
        type=float,
        default=2.0
    )

    parser.add_argument(
        "--nrep",
        type=int,
        default=48
    )

    parser.add_argument(
        "--ntherm",
        type=int,
        default=5000
    )

    parser.add_argument(
        "--nmeas",
        type=int,
        default=10000
    )

    parser.add_argument(
        "--sweeps",
        type=int,
        default=20
    )

    parser.add_argument(
        "--out",
        type=str,
        required=True
    )


    args = parser.parse_args()


    assert args.L % 7 == 0



    print("Building lattice...")


    sites, coords, coord_map, bonds = (
        maple_leaf_lattice_pbc(
            args.L,
            args.L
        )
    )


    bonds = np.array(
        bonds,
        dtype=np.int64
    )


    neighbors = build_neighbors(
        len(sites),
        bonds
    )


    color_sites = color_lattice(
        len(sites),
        bonds
    )


    temps = np.geomspace(
        args.Tmin,
        args.Tmax,
        args.nrep
    )



    print(
        "Starting GPU simulation..."
    )


    result = run_pt_gpu(
        bonds,
        neighbors,
        color_sites,
        temps,
        J=-1.0,
        n_therm=args.ntherm,
        n_meas=args.nmeas,
        sweeps_per_exchange=args.sweeps
    )



    Path(args.out).parent.mkdir(
        parents=True,
        exist_ok=True
    )


    np.savez(
        args.out,
        L=args.L,
        N=len(sites),
        bonds=bonds,
        n_therm=args.ntherm,
        n_meas=args.nmeas,
        sweeps_per_exchange=args.sweeps,
        **result
    )


    print("\n========== DONE ==========")
    print("L =", args.L)
    print("N =", len(sites))



if __name__ == "__main__":
    main()