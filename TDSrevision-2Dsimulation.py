from vpython import *
import numpy as np
import math
import matplotlib.pyplot as plt

# win = 500 # peut aider à définir la taille d'un autre objet visuel comme un histogramme proportionnellement à la taille du canevas.

# Déclaration de variables influençant le temps d'exécution de la simulation
Natoms = 200  # change this to have more or fewer atoms
dt = 1E-5  # pas d'incrémentation temporel

# Déclaration de variables physiques "Typical values"
mass = 4E-3/6E23 # helium mass
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
p = [] # quantité de mouvement des sphères
apos = [] # position des sphères
pavg = sqrt(2*mass*1.5*k*T) # average kinetic energy p**2/(2mass) = (3/2)kT : Principe de l'équipartition de l'énergie en thermodynamique statistique classique # TODO: Changer pour quantités de mouvement initiales aléatoires sur une plage raisonnable cohérente avec température pièce

for i in range(Natoms):
    x = L*random()-L/2 # position aléatoire qui tient compte que l'origine est au centre de la boîte
    y = L*random()-L/2
    z = 0
    if i == 0:  # garde une sphère plus grosse et colorée parmis toutes les grises
        Atoms.append(simple_sphere(pos=vector(x,y,z), radius=0.03, color=color.magenta)) #, make_trail=True, retain=100, trail_radius=0.3*Ratom))
    else: Atoms.append(simple_sphere(pos=vector(x,y,z), radius=Ratom, color=gray))
    apos.append(vec(x,y,z)) # liste de la position initiale de toutes les sphères
#    theta = pi*random() # direction de coordonnées sphériques, superflue en 2D
    phi = 2*pi*random() # direction aléatoire pour la quantité de mouvement
    px = pavg*cos(phi)  # qte de mvt initiale selon l'équipartition
    py = pavg*sin(phi)
    pz = 0
    p.append(vector(px,py,pz)) # liste de la quantité de mouvement initiale de toutes les sphères

#### FONCTION POUR IDENTIFIER LES COLLISIONS, I.E. LORSQUE LA DISTANCE ENTRE LES CENTRES DE 2 SPHÈRES EST À LA LIMITE DE S'INTERPÉNÉTRER ####
def checkCollisions():
    hitlist = []   # initialisation
    r2 = 2*Ratom   # distance critique où les 2 sphères entre en contact à la limite de leur rayon
    r2 *= r2   # produit scalaire pour éviter une comparaison vectorielle ci-dessous
    for i in range(Natoms):
        ai = apos[i]
        for j in range(i) :
            aj = apos[j]
            dr = ai - aj   # la boucle dans une boucle itère pour calculer cette distance vectorielle dr entre chaque paire de sphère
            if mag2(dr) < r2:   # test de collision où mag2(dr) qui retourne la norme élevée au carré de la distance intersphère dr
                hitlist.append([i,j]) # liste numérotant toutes les paires de sphères en collision
    return hitlist

#### BOUCLE PRINCIPALE POUR L'ÉVOLUTION TEMPORELLE DE PAS dt ####
## ATTENTION : la boucle laisse aller l'animation aussi longtemps que souhaité, assurez-vous de savoir comment interrompre vous-même correctement (souvent `ctrl+c`, mais peut varier)
## ALTERNATIVE : vous pouvez bien sûr remplacer la boucle "while" par une boucle "for" avec un nombre d'itérations suffisant pour obtenir une bonne distribution statistique à l'équilibre

for r in range(0,100):
    rate(50)  # limite la vitesse de calcul de la simulation pour que l'animation soit visible à l'oeil humain!

    #### DÉPLACE TOUTES LES SPHÈRES D'UN PAS SPATIAL deltax
    vitesse = []   # vitesse instantanée de chaque sphère
    deltax = []  # pas de position de chaque sphère correspondant à l'incrément de temps dt
    for i in range(Natoms):
        vitesse.append(p[i]/mass)   # par définition de la quantité de nouvement pour chaque sphère
        deltax.append(vitesse[i] * dt)   # différence avant pour calculer l'incrément de position
        Atoms[i].pos = apos[i] = apos[i] + deltax[i]  # nouvelle position de l'atome après l'incrément de temps dt

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
        mtot = 2*mass    # masse totale des 2 sphères
        Vcom = ptot/mtot   # vitesse du référentiel barycentrique/center-of-momentum (com) frame
        posi = apos[i]   # position de chacune des 2 sphères
        posj = apos[j]
        vi = p[i]/mass   # vitesse de chacune des 2 sphères
        vj = p[j]/mass
        rrel = posi-posj  # vecteur pour la distance entre les centres des 2 sphères
        vrel = vj-vi   # vecteur pour la différence de vitesse entre les 2 sphères

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
        pcomi = p[i]-mass*Vcom  # transform momenta to center-of-momentum (com) frame
        pcomj = p[j]-mass*Vcom
        rrel = hat(rrel)    # vecteur unitaire aligné avec rrel
        pcomi = pcomi-2*dot(pcomi,rrel)*rrel # bounce in center-of-momentum (com) frame
        pcomj = pcomj-2*dot(pcomj,rrel)*rrel  # TODO: convertir avec masse réduite et vitesse du centre de masse en corrigeant les unités
        p[i] = pcomi+mass*Vcom # transform momenta back to lab frame
        p[j] = pcomj+mass*Vcom
        apos[i] = posi+(p[i]/mass)*deltat # move forward deltat in time, ramenant au même temps où sont rendues les autres sphères dans l'itération
        apos[j] = posj+(p[j]/mass)*deltat

#Question 1

dotp=[] #liste vide qui va contenir le produit scalaire de tous les vecteurs de la liste entre eux-mêmes
for i in range(len(p)):
    dotp.append(dot(p[i],p[i]))

dotp_moy=sum(dotp)/len(dotp)

print('Moyenne du carré de la quantité de mouvement: '+str(dotp_moy)+' kg*m/s')

#Question 2

print('Température retrouvée: '+str(round((2*dotp_moy)/(3*2*mass*k),5))+' K')

#------------------------------------

#Question 3

def particule_seule():
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
    p = [] # quantité de mouvement des sphères
    apos = [] # position des sphères
    pavg = sqrt(2*mass*1.5*k*T) # average kinetic energy p**2/(2mass) = (3/2)kT : Principe de l'équipartition de l'énergie en thermodynamique statistique classique # TODO: Changer pour quantités de mouvement initiales aléatoires sur une plage raisonnable cohérente avec température pièce

    for i in range(Natoms):
        x = L*random()-L/2 # position aléatoire qui tient compte que l'origine est au centre de la boîte
        y = L*random()-L/2
        z = 0
        if i == 0:  # garde une sphère plus grosse et colorée parmis toutes les grises
            Atoms.append(simple_sphere(pos=vector(x,y,z), radius=0.03, color=color.magenta)) #, make_trail=True, retain=100, trail_radius=0.3*Ratom))
        else: Atoms.append(simple_sphere(pos=vector(x,y,z), radius=Ratom, color=gray))
        apos.append(vec(x,y,z)) # liste de la position initiale de toutes les sphères
    #    theta = pi*random() # direction de coordonnées sphériques, superflue en 2D
        phi = 2*pi*random() # direction aléatoire pour la quantité de mouvement
        px = pavg*cos(phi)  # qte de mvt initiale selon l'équipartition
        py = pavg*sin(phi)
        pz = 0
        p.append(vector(px,py,pz)) # liste de la quantité de mouvement initiale de toutes les sphères


        #### FONCTION POUR IDENTIFIER LES COLLISIONS, I.E. LORSQUE LA DISTANCE ENTRE LES CENTRES DE 2 SPHÈRES EST À LA LIMITE DE S'INTERPÉNÉTRER ####
    def checkCollisions_seule(r,iteration_coll):
        hitlist = []   # initialisation
        r2 = 2*Ratom   # distance critique où les 2 sphères entre en contact à la limite de leur rayon
        r2 *= r2   # produit scalaire pour éviter une comparaison vectorielle ci-dessous
        for i in range(Natoms):
            ai = apos[i]
            for j in range(i) :
                aj = apos[j]
                dr = ai - aj   # la boucle dans une boucle itère pour calculer cette distance vectorielle dr entre chaque paire de sphère
                if mag2(dr) < r2:   # test de collision où mag2(dr) qui retourne la norme élevée au carré de la distance intersphère dr
                    hitlist.append([i,j]) # liste numérotant toutes les paires de sphères en collision
        if (", 0]" in ''.join(str(i) for i in hitlist))==True: # une fois toutes les collisions répertoriées, vérifie si l'atome 0 a subit au moins une collision dans l'itération r. Si oui, on l'ajoute à la liste des itérations collisionnelles.
            iteration_coll.append(r)
        return hitlist,iteration_coll # retourne un tuple comprenant la liste des collisions totale et la liste des itérations pour lesquels au moins une collision de l'atome 0 est enregistrée.

    #### BOUCLE PRINCIPALE POUR L'ÉVOLUTION TEMPORELLE DE PAS dt ####
    ## ATTENTION : la boucle laisse aller l'animation aussi longtemps que souhaité, assurez-vous de savoir comment interrompre vous-même correctement (souvent `ctrl+c`, mais peut varier)
    ## ALTERNATIVE : vous pouvez bien sûr remplacer la boucle "while" par une boucle "for" avec un nombre d'itérations suffisant pour obtenir une bonne distribution statistique à l'équilibre

    #Objectif: suivre l'atome 0
    iteration_coll=[] # iterations à lesquelles l'atome 0 subit une collision
    iteration_vitesse=[] # vitesse de la particule 0 à chaque itération  

    for r in range(200): #nombre d'itération de pas dt
        rate(25)  # limite la vitesse de calcul de la simulation pour que l'animation soit visible à l'oeil humain!

        #### DÉPLACE TOUTES LES SPHÈRES D'UN PAS SPATIAL deltax
        vitesse = []   # vitesse instantanée de chaque sphère
        deltax = []  # pas de position de chaque sphère correspondant à l'incrément de temps dt
        for i in range(Natoms):
            vitesse.append(p[i]/mass)   # par définition de la quantité de nouvement pour chaque sphère
            deltax.append(vitesse[i] * dt)   # différence avant pour calculer l'incrément de position
            Atoms[i].pos = apos[i] = apos[i] + deltax[i]  # nouvelle position de l'atome après l'incrément de temps dt
            if i==0:
                iteration_vitesse.append(vitesse[i])

        #### CONSERVE LA QUANTITÉ DE MOUVEMENT AUX COLLISIONS AVEC LES MURS DE LA BOÎTE ####
        for i in range(Natoms):
            loc = apos[i]
            if abs(loc.x) > L/2:
                if loc.x < 0: p[i].x =  abs(p[i].x)  # renverse composante x au mur de gauche
                else: p[i].x =  -abs(p[i].x)   # renverse composante x au mur de droite
                if i==0:
                    iteration_coll.append(r) #stockage de l'itération à laquelle la sphère rose frappe un mur (gauche ou droite)
            if abs(loc.y) > L/2:
                if loc.y < 0: p[i].y = abs(p[i].y)  # renverse composante y au mur du bas
                else: p[i].y =  -abs(p[i].y)  # renverse composante y au mur du haut
                if i==0:
                    iteration_coll.append(r) #stockage de l'itération à laquelle la sphère rose frappe un mur (haut ou bas)
                    
        #### LET'S FIND THESE COLLISIONS!!! ####
        checkcollision_tuple = checkCollisions_seule(r,iteration_coll)
        hitlist = checkcollision_tuple[0]
        iteration_coll=checkcollision_tuple[1]

        #### CONSERVE LA QUANTITÉ DE MOUVEMENT AUX COLLISIONS ENTRE SPHÈRES ####
        for ij in hitlist:

            # définition de nouvelles variables pour chaque paire de sphères en collision
            i = ij[0]  # extraction du numéro des 2 sphères impliquées à cette itération
            j = ij[1]
            ptot = p[i]+p[j]   # quantité de mouvement totale des 2 sphères
            mtot = 2*mass    # masse totale des 2 sphères
            Vcom = ptot/mtot   # vitesse du référentiel barycentrique/center-of-momentum (com) frame
            posi = apos[i]   # position de chacune des 2 sphères
            posj = apos[j]
            vi = p[i]/mass   # vitesse de chacune des 2 sphères
            vj = p[j]/mass
            rrel = posi-posj  # vecteur pour la distance entre les centres des 2 sphères
            vrel = vj-vi   # vecteur pour la différence de vitesse entre les 2 sphères

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
            pcomi = p[i]-mass*Vcom  # transform momenta to center-of-momentum (com) frame
            pcomj = p[j]-mass*Vcom
            rrel = hat(rrel)    # vecteur unitaire aligné avec rrel
            pcomi = pcomi-2*dot(pcomi,rrel)*rrel # bounce in center-of-momentum (com) frame
            pcomj = pcomj-2*dot(pcomj,rrel)*rrel  # TODO: convertir avec masse réduite et vitesse du centre de masse en corrigeant les unités
            p[i] = pcomi+mass*Vcom # transform momenta back to lab frame
            p[j] = pcomj+mass*Vcom
            apos[i] = posi+(p[i]/mass)*deltat # move forward deltat in time, ramenant au même temps où sont rendues les autres sphères dans l'itération
            apos[j] = posj+(p[j]/mass)*deltat


    print(iteration_coll)
    nb_iteration_entre_coll = [j-i for i, j in zip(iteration_coll[:-1], iteration_coll[1:])] # nombre d'itérations entre deux itérations correspondantes à une collision de l'atome 0
    print(nb_iteration_entre_coll)
    temps_entre_coll = [i * dt for i in nb_iteration_entre_coll]

    vitesse_coll_mag = [mag(iteration_vitesse[i]) for i in iteration_coll]
    vitesse_coll_vecteur = [iteration_vitesse[i] for i in iteration_coll]
    deplacement_entre_coll = [i * j for i,j in zip(vitesse_coll_mag, temps_entre_coll)]

    temps_moy=sum(temps_entre_coll)/len(temps_entre_coll)
    deplacement_moy=sum(deplacement_entre_coll)/len(deplacement_entre_coll)

    return deplacement_entre_coll,temps_entre_coll, deplacement_moy, temps_moy, vitesse_coll_mag, vitesse_coll_vecteur



#Question 4

resultats_particule_seule = particule_seule()

print('Libre parcours moyen de la particule seule: '+str(round(resultats_particule_seule[2],5))+' m')
print('Temps moyen entre les collisions de la particule seule: '+str(round(resultats_particule_seule[3],5))+' s')



#Question 5

vitesse_entre_coll_seule = [i/j for i, j in zip(resultats_particule_seule[0], resultats_particule_seule[1])] # v=x/t pour chaque "entre collisions"
vitesse_moyenne_particule_seule = sum(vitesse_entre_coll_seule)/len(vitesse_entre_coll_seule)
print('Vitesse moyenne de la particule seule: '+str(round(vitesse_moyenne_particule_seule,5))+' m/s')


#Question 6

vitesse_entre_coll_mag_seule=resultats_particule_seule[4]
vitesse_entre_coll_vecteur_seule=resultats_particule_seule[5]
vitesse_entre_coll_carre_seule=[dot(vitesse_entre_coll_vecteur_seule[i],vitesse_entre_coll_vecteur_seule[i]) for i in range(0,len(vitesse_entre_coll_vecteur_seule))]

vitesse_entre_coll_x_seule = [vitesse_entre_coll_vecteur_seule[i].x for i in range(len(vitesse_entre_coll_vecteur_seule))]

#print(vitesse_entre_coll_mag_seule)
#print(vitesse_entre_coll_carre_seule)
#print(vitesse_entre_coll_x_seule)

fig1=plt.figure()

fig1, axs = plt.subplots(3, 1)

axs[0].hist(vitesse_entre_coll_mag_seule, bins=20, edgecolor='black', color='red')
axs[0].title.set_text('Module de la vitesse ||v||')

axs[1].hist(vitesse_entre_coll_carre_seule, bins=20, edgecolor='black', color='green')
axs[1].title.set_text('Module de la vitesse au carré ||v^2||')
axs[1].set_ylabel("Quantité d'inter collisions")

axs[2].hist(vitesse_entre_coll_x_seule, bins=20, edgecolor='black', color='blue')
axs[2].title.set_text('Module de la vitesse en x ||v_x||')
axs[2].set_xlabel("Vitesse [m/s]")

plt.tight_layout()

#fig1.savefig("/Users/david.tremblay7/Desktop/histo_vitesses.pdf", bbox_inches='tight')

plt.show()