#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 15:40:27 2021

#GlowScript 3.0 VPython

# Hard-sphere gas.

# Bruce Sherwood
# Claudine Allen
"""

from vpython import *
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
from scipy.stats import maxwell

# win = 500 # peut aider à définir la taille d'un autre objet visuel comme un histogramme proportionnellement à la taille du canevas.

# Déclaration de variables influençant le temps d'exécution de la simulation
Natoms = 100  # change this to have more or fewer atoms
Ncoeurs = 9 # Nb de coeurs (doit être un nombre impair au carré pour produire un nombre périodique de coeurs)
dt = 1E-8  # pas d'incrémentation temporel

# Déclaration de variables physiques "Typical values"
mass_electron = 9.11*10**(-31) # masse électron en kg
Ratom = 0.01 # wildly exaggerated size of an atom
k = 1.4E-23 # Boltzmann constant # TODO: changer pour une constante de Boltzmann en eV/K :)
T = 300 # around room temperature

#### CANEVAS DE FOND ####
L = 1 # container is a cube L on a side
gray = color.gray(0.7) # color of edges of container and spheres below
animation = canvas( width=750, height=500) # , align='left')
animation.range = L
# animation.title = 'Théorie cinétique des gaz parfaits'
# s = """  Simulation de particules modélisées en sphères dures pour représenter leur trajectoire ballistique avec collisions. Une sphère est colorée et grossie seulement pour l’effet visuel permettant de suivre sa trajectoire plus facilement dans l'animation, sa cinétique est identique à toutes les autres particules.

# """
# animation.caption = s

#### ARÊTES DE BOÎTE 2D ####
d = L/2+Ratom
r = 0.005
cadre = curve(color=gray, radius=r)
cadre.append([vector(-d,-d,0), vector(d,-d,0), vector(d,d,0), vector(-d,d,0), vector(-d,-d,0)])

#### POSITION ET QUANTITÉ DE MOUVEMENT INITIALE DES SPHÈRES ####
Atoms = [] # Objet qui contiendra les sphères pour l'animation
Coeurs = []
p = [] # quantité de mouvement des sphères
apos = [] # position des électrons
coeurs_pos = [] # Position des coeurs
vecteur_vitesse = np.linspace(0,300000,10000) # intervalle des vitesses en m/s (valeurs de vitesse que nos électrons peuvent prendre)
qte_mouv_vecteur = mass_electron * vecteur_vitesse

# on crée une fonction pour la distribution de maxwell boltzmann (voir référence), on ne fait que définir la fonction pour que nos particules
# suivent cette distribution comme demandé
def m_b(k,T,m,v):
    fonction = (m / (2 * np.pi * k * T)) ** (3 / 2) * 4 * np.pi * v ** 2 * np.exp(-m * v ** 2 / (2 * k * T))
    fonction = fonction/sum(fonction) # on normalise pour que l'intégrale donne 1, ça facilite et donne une meilleure représentation
    return fonction
fonction_m_b = m_b(k,T,mass_electron,vecteur_vitesse)

# On place les électrons
for i in range(Natoms):
    x = L*random()-L/2 # position aléatoire qui tient compte que l'origine est au centre de la boîte
    y = L*random()-L/2
    z = 0
    if i == 0:  # garde une sphère plus grosse et colorée parmis toutes les grises
        Atoms.append(simple_sphere(pos=vector(x,y,z), radius=0.03, color=color.magenta)) #, make_trail=True, retain=100, trail_radius=0.3*Ratom))

    else: Atoms.append(simple_sphere(pos=vector(x,y,z), radius=Ratom, color=gray))
    qte_mouv = np.random.choice(a = qte_mouv_vecteur, size = 1, p = fonction_m_b) # on fixe la quantité de mouvement afin
    # que nos électrons suivent la distribution de maxwell boltzmann comme demandé
    phi = 2*pi*random() # direction aléatoire pour la quantité de mouvement
    apos.append(vec(x,y,z)) # liste de la position initiale de toutes les sphères
    px = qte_mouv*cos(phi)  # qte de mvt initiale selon l'équipartition (suivant la distribution de maxwell_boltzmann)
    py = qte_mouv*sin(phi)
    pz = 0
    p.append(vector(px,py,pz)) # liste de la quantité de mouvement initiale de toutes les sphères

# On place les coeurs immobile dans la boite, de cette manière, les coeurs se placent automatiquement en changeant la valeur de la 
# variable "Ncoeurs".  
x = []
for i in range(int(-(sqrt(Ncoeurs) - 1)/2), int((sqrt(Ncoeurs) -1)/2) + 1):
    x.append(i*L/(sqrt(Ncoeurs) + 1))
y = []
for i in range(int(-(sqrt(Ncoeurs) - 1)/2), int((sqrt(Ncoeurs) -1)/2) + 1):
    y.append(i*L/(sqrt(Ncoeurs) + 1))
z = 0
for i in x:
    for c in y:
        Coeurs.append(simple_sphere(pos=vector(i,c,0), radius=0.02, color=color.yellow))
        apos.append(vec(i,c,0))


#### FONCTION POUR IDENTIFIER LES COLLISIONS, I.E. LORSQUE LA DISTANCE ENTRE LES CENTRES DE 2 SPHÈRES EST À LA LIMITE DE S'INTERPÉNÉTRER ####
def checkCollisions():
    hitlist = []   # initialisation
    r2 = 2*Ratom   # distance critique où les 2 sphères entre en contact à la limite de leur rayon
    r2 *= r2   # produit scalaire pour éviter une comparaison vectorielle ci-dessous
    for i in range(Natoms + Ncoeurs):
        ai = apos[i]
        for j in range(i) : 
            aj = apos[j] 
            dr = ai - aj   # la boucle dans une boucle itère pour calculer cette distance vectorielle dr entre chaque paire de sphère
            if mag2(dr) < r2:   # test de collision où mag2(dr) qui retourne la norme élevée au carré de la distance intersphère dr
                if i > Natoms: # On enlève les collisions entre les électrons
                    hitlist.append([i,j]) # Ici, on enregistre la sphère et rien d'autre, on prend le min car des indices de liste
                    # doivent être des integers et non des listes, ça ne change pas l'animation, juste le nom de la collision dans la hitlist
    return hitlist


#### BOUCLE PRINCIPALE POUR L'ÉVOLUTION TEMPORELLE DE PAS dt ####
## ATTENTION : la boucle laisse aller l'animation aussi longtemps que souhaité, assurez-vous de savoir comment interrompre vous-même correctement (souvent `ctrl+c`, mais peut varier)
## ALTERNATIVE : vous pouvez bien sûr remplacer la boucle "while" par une boucle "for" avec un nombre d'itérations suffisant pour obtenir une bonne distribution statistique à l'équilibre

for r in range(0,100):
    rate(150)  # limite la vitesse de calcul de la simulation pour que l'animation soit visible à l'oeil humain!

    #### DÉPLACE TOUTES LES SPHÈRES D'UN PAS SPATIAL deltax
    vitesse = []   # vitesse instantanée de chaque sphère
    deltax = []  # pas de position de chaque sphère correspondant à l'incrément de temps dt
    for i in range(Natoms):
        vitesse.append(p[i]/mass_electron)   # par définition de la quantité de nouvement pour chaque sphère
        deltax.append(vitesse[i] * dt)   # différence avant pour calculer l'incrément de position
        Atoms[i].pos = apos[i] = apos[i] + deltax[i]  # nouvelle position de l'atome après l'incrément de temps dt
        print(vitesse)
        print(deltax)
    #### CONSERVE LA QUANTITÉ DE MOUVEMENT AUX COLLISIONS AVEC LES MURS DE LA BOÎTE ####
    for i in range(Natoms):
        loc = apos[i]
        if abs(loc.x) > L/2:
            if loc.x < 0: p[i].x =  abs(p[i].x)  # renverse composante x au mur de gauche
            else: p[i].x =  -abs(p[i].x)   # renverse composante x au mur de droite
        if abs(loc.y) > L/2:
            if loc.y < 0: p[i].y = abs(p[i].y)  # renverse composante y au mur du bas
            else: p[i].y =  -abs(p[i].y)  # renverse composante y au mur du haut

    #### LET'S FIND THESE COLLISIONS!!! ####
    hitlist = checkCollisions()

    #### CONSERVE LA QUANTITÉ DE MOUVEMENT AUX COLLISIONS ENTRE SPHÈRES ####
    for ij in hitlist:
        # définition de nouvelles variables pour chaque paire de sphères en collision
        i = ij[0]  # extraction du numéro des 2 sphères impliquées à cette itération
        j = ij[1]
        ptot = p[i]+p[j]   # quantité de mouvement totale des 2 sphères
        mtot = 2*mass_electron    # masse totale des 2 sphères
        Vcom = ptot/mtot   # vitesse du référentiel barycentrique/center-of-momentum (com) frame
        posi = apos[i]   # position de chacune des électrons
        posj = coeurs_pos[j] # On change "apos[j]" pour "coeurs_pos[j]" pour que les collisions ne se fassent qu'avec les coeurs
        vi = p[i]/mass_electron   # vitesse de chacune des 2 sphères
        vj = p[j]/mass_electron
        rrel = posi-posj  # vecteur pour la distance entre les centres des 2 sphères
        vrel = -vi   # vecteur pour la différence de vitesse entre les 2 sphères

        # exclusion de cas où il n'y a pas de changements à faire
        if vrel.mag2 == 0: continue  # exactly same velocities si et seulement si le vecteur vrel devient nul, la trajectoire des 2 sphères continue alors côte à côte
        if rrel.mag > Ratom: continue  # one atom went all the way through another, la collision a été "manquée" à l'intérieur du pas deltax # TODO: ajouter une boucle fractionnant dt à une limite raisonnable pour aller chercher la collision manquée

        # calcule la distance et temps d'interpénétration des sphères dures qui ne doit pas se produire dans ce modèle # TODO: donner géométrie en devoir en illustrant avec Geogebra -> Tikz et/ou animation wiki https://en.wikipedia.org/wiki/Elastic_collision#Two-dimensional_collision_with_two_moving_objects
        dx = dot(rrel, vrel.hat)       # rrel.mag*cos(theta) où theta is the angle between vrel and rrel:
        dy = cross(rrel, vrel.hat).mag # rrel.mag*sin(theta)
        alpha = asin(dy/(2*Ratom))  # alpha is the angle of the triangle composed of rrel, path of atom j, and a line from the center of atom i to the center of atom j where atome j hits atom i
        d = (2*Ratom)*cos(alpha)-dx # distance traveled into the atom from first contact
        deltat = d/vrel.mag         # time spent moving from first contact to position inside atom

        #### CHANGE L'INTERPÉNÉTRATION DES SPHÈRES PAR LA CINÉTIQUE DE COLLISION ####
        posi = posi-vi*deltat   # back up to contact configuration
        posj = posj-vj*deltat
        pcomi = p[i]-mass_electron*Vcom  # transform momenta to center-of-momentum (com) frame
        pcomj = -mass_electron*Vcom
        rrel = hat(rrel)    # vecteur unitaire aligné avec rrel

        pcomi = pcomi-2*dot(pcomi,rrel)*rrel # bounce in center-of-momentum (com) frame
        pcomj = pcomj-2*dot(pcomj,rrel)*rrel  # TODO: convertir avec masse réduite et vitesse du centre de masse en corrigeant les unités
        p[i] = pcomi+mass_electron*Vcom # transform momenta back to lab frame
        p[j] = pcomj+mass_electron*Vcom
        apos[i] = posi+(p[i]/mass_electron)*deltat # move forward deltat in time, ramenant au même temps où sont rendues les autres sphères dans l'itération
        # On change "apos[j]" pour "coeurs_pos[j]" pour que les collisions ne se fassent qu'avec les coeurs
        apos[j] = posj+(p[j]/mass_electron)*deltat

    


# On veut maintenant ajouter un champ électrique uniforme perpendiculaire à deux cotés de la boîte. Pour ce faire, on va tout simplement
# créer une fonction qui prendra comme argument la norme et l'orientation du champ afin de retourner une certaine valeur de delta_quantité_mouv
# en x et en y.

#def Champ_electrique(orientation,norme):
   # champ_elec_x = norme*cos(orientation*(pi/2))
   # champ_elec_y = norme*sin(orientation*(pi/2))
  #  q = 1.602*10**(-19)
   # delta_qte_mouv_x = -q*champ_elec_x*dt
  #  delta_qte_mouv_y = -q*champ_elec_y*dt
   # return delta_qte_mouv_x, delta_qte_mouv_y