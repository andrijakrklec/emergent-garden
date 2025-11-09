"""
AUTHOR: Vishal Paudel
(Modificirano za CFL simulaciju)
"""
from typing import Tuple, List
from ctypes import c_ubyte

import pygame
import random
import math
import numpy as np
from sklearn.cluster import KMeans


from src.constants import (
    PARTICLE_DEFAULT_RADIUS, SCREEN_DIM, WALL_HEAT, WALL_BOUNDARY,
    PARTICLE_FORCE_LOWER_RANGE, PARTICLE_FORCE_UPPER_RANGE,
    PARTICLE_POWER_OF_DISTANCE, PARTICLE_DEFAULT_UPDATE_TIME, PARTICLE_LOSE_ENERGY
)

class Particle:
    def __init__(
        self, x: Tuple[int, int],
        v: Tuple[float, float],
        c: Tuple[c_ubyte, c_ubyte, c_ubyte],
        r: float = PARTICLE_DEFAULT_RADIUS,
        target_pos: Tuple[int, int] = (0, 0) # NOVO: Skriveni cilj (lokalni podaci)
        ):
        """Initializes a particle

        Args:
            x   (Tuple[int, int]):      The postion     vector  of the particle
            v   (Tuple[float, float]):  The velocity    vector  of the particle
            c   (Tuple[int, int, int]): The color       RGB     of the particle
            r   (int):                  The radius      pixel   of the particle
            target_pos (Tuple[int, int]): The "hidden" target for this particle (its "local data")
        """
        # The position attributes
        self.x = x[0]
        self.y = x[1]

        # The velocity attributes
        self.vx = v[0]
        self.vy = v[1]

        # The look and feel attributes
        self.c = c
        self.r = r

        # --- NOVO: CFL Atributi ---
        self.target_pos = target_pos # "Lokalni podaci"

        # "Lokalni model" - Inicijaliziramo ga kao nasumični 2D vektor
        self.model = np.random.rand(2) * 2 - 1
        self.model = self.model / np.linalg.norm(self.model) # Normaliziramo

        self.cluster_id = -1 # Pripadnost klasteru (-1 = nije dodijeljen)


    def update(self, dt: float):
        """Updates the attributes of the particle

        Args:
            dt (float): The delta time, time since the last frame update
        """
        # (Ova funkcija se trenutno ne koristi, fizika se rješava centralno)
        pass


    def draw(self, screen: pygame.Surface):
        """Draws the particle(a circle)

        Args:
            screen (pygame.Surface):    The     screen  to draw onto
        """
        pygame.draw.circle(screen, self.c, (self.x, self.y), self.r)


def local_train(particle: Particle, learning_rate: float = 0.1):
    """
    Simulira 1 korak lokalnog treninga.
    Čestica ažurira svoj "model" (svoj vektor smjera)
    kako bi bolje odgovarao njenom "skrivenom cilju" (lokalni podaci).
    """
    # Izračunaj "idealan" smjer prema cilju
    target_vec = np.array([
        particle.target_pos[0] - particle.x,
        particle.target_pos[1] - particle.y
    ])

    # Izbjegni dijeljenje s nulom ako je već na cilju
    norm = np.linalg.norm(target_vec)
    if norm > 0:
        target_vec = target_vec / norm # Normaliziraj "gradijent"

    # Ažuriraj model (gradijentni uspon)
    particle.model = particle.model + learning_rate * target_vec

    # Ponovno normaliziraj model da ostane jedinični vektor
    norm_model = np.linalg.norm(particle.model)
    if norm_model > 0:
        particle.model = particle.model / norm_model

def run_cfl_round(particles: List[Particle], kmeans_model: KMeans):
    """
    Pokreće jedan puni krug CFL-a:
    1. Skuplja sve "modele" (lokalne težine).
    2. Grupira modele koristeći KMeans.
    3. Izračunava novi agregirani model za svaki klaster.
    4. Emitira nove modele i cluster_id natrag svakoj čestici.
    """

    # 1. Skupljanje modela
    all_models = np.array([p.model for p in particles])

    # 2. Grupiranje modela
    labels = kmeans_model.fit_predict(all_models)

    # 3. Agregacija (centri klastera su naši novi agregirani modeli)
    cluster_centers = kmeans_model.cluster_centers_

    # 4. Emitiranje
    for i, particle in enumerate(particles):
        particle.cluster_id = labels[i]

        # Postavi model čestice na novi, agregirani model njenog klastera
        # Normaliziraj ga za svaki slučaj
        new_model = cluster_centers[particle.cluster_id]
        norm = np.linalg.norm(new_model)
        if norm > 0:
            particle.model = new_model / norm
        else:
            particle.model = new_model

def apply_physics_rules(particles: List[Particle], g_attract: float, g_repel: float, dt: float = PARTICLE_DEFAULT_UPDATE_TIME):
    """
    Primjenjuje fiziku privlačenja/odbijanja temeljenu ISKLJUČIVO na cluster_id.
    """

    # Pohrani sile prije primjene da se izbjegnu konflikti
    forces = [np.zeros(2) for _ in particles]

    for i in range(len(particles)):
        a = particles[i]
        for j in range(i + 1, len(particles)): # Optimizacija: i+1
            b = particles[j]

            dx = a.x - b.x
            dy = a.y - b.y
            d_sq = dx**2 + dy**2

            # Preskoči ako su čestice preklopljene
            if d_sq == 0:
                continue

            d = d_sq**0.5

            if (PARTICLE_FORCE_UPPER_RANGE > d > PARTICLE_FORCE_LOWER_RANGE):
                g = 0.0  # Zadana vrijednost (nema sile)

                # Prvo provjeri jesu li obje čestice uopće grupirane
                if a.cluster_id == -1 or b.cluster_id == -1:
                    g = 0.0  # Ne primjenjuj silu ako ijedna nije klasificirana

                # Ako jesu, primijeni pravila
                elif a.cluster_id == b.cluster_id:
                    g = g_attract  # U istom klasteru -> PRIVLAČENJE

                else:  # (a.cluster_id != b.cluster_id)
                    g = g_repel

                F_scalar = g * (1 / (d ** PARTICLE_POWER_OF_DISTANCE * len(particles)))

                fx = F_scalar * dx
                fy = F_scalar * dy

                forces[i] += np.array([fx, fy])
                forces[j] -= np.array([fx, fy]) # Newtonov 3. zakon

    # Sada primijeni izračunate sile i ažuriraj pozicije
    for i, a in enumerate(particles):
        fx, fy = forces[i]

        # AŽURIRANJE BRZINE I POZICIJE
        a.vx = (a.vx + fx * dt) * PARTICLE_LOSE_ENERGY
        a.vy = (a.vy + fy * dt) * PARTICLE_LOSE_ENERGY

        a.x += a.vx * dt
        a.y += a.vy * dt

        V = 0.9  #was WALL_HEAT
        D = WALL_BOUNDARY
        if(a.x < D):
            a.x = D + 1
            a.vx *= -V
        elif(a.x > SCREEN_DIM[0] - D):
            a.x = SCREEN_DIM[0] - D - 1
            a.vx *= -V

        if(a.y < D):
            a.y = D + 1
            a.vy *= -V
        elif(a.y > SCREEN_DIM[1] - D):
            a.y = SCREEN_DIM[1] - D - 1
            a.vy *= -V


def instantiateGroup(
    num: int,
    c: tuple,
    frame: Tuple[Tuple[int, int], Tuple[int, int]],
    target_pos: Tuple[int, int] # Dodan target_pos
    ) -> List[Particle]:
    """Method to instantiate a group of particles

    Args:
        num     (int):                                      The number          of particles in the group
        c       (tuple):                                    The color           of the group
        frame   (Tuple[Tuple[int, int], Tuple[int, int]]):  The cordinate frame for the group to spawn in
        target_pos (Tuple[int, int]):                       The hidden target   for this group

    Returns:
        group   List[Particle]:                             The list    of particles        in the group
    """
    random.seed()

    group = []
    for _ in range(num):
        x = random.randint(frame[0][0], frame[0][1])
        y = random.randint(frame[1][0], frame[1][1])

        # Proslijedi target_pos u konstruktor
        group.append(Particle(x=(x, y), v=(0.0, 0.0), c=c, target_pos=target_pos))

    return group