# coding=utf-8
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import math as m

def busca_dist_menor(m):
    elem_min = m[0, 1:].min()
    filas, columnas = m.shape
    encontrado = False
    colum = 1
    while colum < columnas and not encontrado:
        if elem_min == m[0, colum]:
            encontrado = True
        else:
            colum += 1
    return colum


def argmin_diagonal_ignored(m):
    mask = np.ones(m.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    elem_min = m[mask].min()
    fila = 0
    filas, columnas = m.shape
    encontrado = False
    while fila < filas and not encontrado:
        columna = fila + 1
        while columna < columnas and not encontrado:
            if elem_min == m[fila, columna]:
                encontrado = True
            columna += 1
        fila += 1
    return fila - 1, columna - 1


def obten_num_ptos(list_nptos, ncentroides):
    num_ptos = 0
    for elem in list_nptos:
        if elem > ncentroides:
            num_ptos+=ncentroides
        else:
            num_ptos+=elem
    return num_ptos

def divide_en_grupos(vector, longitud, ngrupos, tam):
    # longitud, _ = vector.shape
    if longitud == ngrupos*tam:
        # La división es exacta:
        vector = np.array(np.split(vector,ngrupos))
    else:
        # La división no es exacta:
        v1 = vector[:tam*(ngrupos-1)]
        resto = vector[tam*(ngrupos-1):]
        v1 = np.split(v1, ngrupos-1)
        v1.append(resto)
        vector = np.array(v1)
        #vector = np.concatenate([v1, resto])

    return vector

def obten_idgrupo(id_punto, grupos):
    ngrupos = len(grupos)
    id_grupo = 0
    inicio = 0
    fin = grupos[0]
    encontrado = False
    while (id_grupo < ngrupos) and not(encontrado):
        if id_punto>=inicio and id_punto<fin:
            encontrado = True
        else :
            id_grupo += 1
            inicio = fin
            fin = fin+grupos[id_grupo]

    return id_grupo


### OPERACIONES PARA LA REAGRUPACIÓN DE GRUPOS DESPUÉS DE LA 1º DECONSTRUCCIÓN ###

def myFunc(e):
  return len(e)

def busca_candidato(D, id_grupo):
    '''filaini = D[id_grupo, 0:id_grupo]
    filafin = D[id_grupo, (id_grupo+1):]
    if len(filaini) == 0:
        fila = filafin
    elif len(filafin) == 0:
        fila = filaini
    else:
        fila = filaini + filafin
    elem_min = fila.min()
    fila = fila.tolist()
    indice = fila.index(elem_min)'''
    fila = D[id_grupo, : ].tolist()
    filaOrd = sorted(fila)
    minimo = filaOrd[1]
    indice = fila.index(minimo)
    return indice

def hay_grupos_peq(vector, tam_grupo):
    bandera = False
    cont = 0
    for elem in vector:
        if (len(elem) < tam_grupo):
            cont += 1

    if cont == len(vector):
        bandera = True

    return bandera

def funcdist(punto, vector, dim):
    vecdist = np.empty(len(vector), float)
    for i in range(len(vector)):
        suma = 0.0
        for n in range(dim):
            difcoord = (punto[0][n]-vector[i][n])*(punto[0][n]-vector[i][n])
            suma += difcoord
        vecdist[i] = m.sqrt(suma)
    return vecdist


def calculate_numcapas(cant_ptos, tam_grupo, n_centroides):
    if cant_ptos < tam_grupo or tam_grupo == n_centroides:
        ncapas = 1
    else:
        cociente = int(cant_ptos / tam_grupo)
        resto = cant_ptos % tam_grupo
        grupos = cociente + resto
        new_ptos = grupos * n_centroides
        ncapas = 1
        while new_ptos > n_centroides:
            cociente = int(new_ptos / tam_grupo)
            resto = new_ptos % tam_grupo
            if resto == 0:
                grupos = cociente
                new_ptos = grupos * n_centroides
            elif resto < n_centroides:
                new_ptos = (cociente * n_centroides) + resto
                grupos = cociente + 1
            elif resto >= n_centroides:
                grupos = cociente + 1
                new_ptos = grupos * n_centroides

            if new_ptos >= n_centroides:
                ncapas += 1

    return ncapas


def built_estructuras_capa(cant_ptos, tam_grupo, n_centroides, n_capas, dimensiones):
    labels_capa = np.empty(n_capas, object)
    puntos_capa = np.empty(n_capas, object)
    grupos_capa = np.empty(n_capas, object)

    # Numero de grupos de la capa 0
    ngrupos = int(cant_ptos / tam_grupo)
    resto = cant_ptos % tam_grupo

    for capa in range(n_capas):

        if resto != 0:
            # resto = cant_ptos - (ngrupos * tam_grupo)
            ngrupos = ngrupos + 1
            labels_grupo = np.empty(ngrupos, object)
            for num in range(ngrupos - 1):
                labels_grupo[num] = np.zeros(tam_grupo, dtype=int)
            labels_grupo[ngrupos - 1] = np.zeros(resto, dtype=int)
            labels_capa[capa] = labels_grupo
            if (resto >= n_centroides):
                puntos_grupo = np.zeros((ngrupos, n_centroides, dimensiones), dtype=float)
                resto_nuevo = (ngrupos * n_centroides) % tam_grupo
                ngrupos_nuevo = int((ngrupos * n_centroides) / tam_grupo)
            else:
                puntos_grupo = np.empty(ngrupos, object)
                for num in range(ngrupos - 1):
                    puntos_grupo[num] = np.zeros((ngrupos - 1, n_centroides, dimensiones))
                puntos_grupo[ngrupos - 1] = np.zeros((1, resto, dimensiones))
                resto_nuevo = ((ngrupos - 1) * n_centroides + resto) % tam_grupo
                ngrupos_nuevo = int(((ngrupos - 1) * n_centroides + resto) / tam_grupo)
            puntos_capa[capa] = puntos_grupo
            grupos_capa[capa] = np.zeros(ngrupos, dtype=int)
        else:
            puntos_capa[capa] = np.zeros((ngrupos, n_centroides, dimensiones), dtype=float)
            labels_capa[capa] = np.zeros((ngrupos, tam_grupo), dtype=int)
            grupos_capa[capa] = np.zeros(ngrupos, dtype=int)
            resto_nuevo = (ngrupos * n_centroides) % tam_grupo
            ngrupos_nuevo = int((ngrupos * n_centroides) / tam_grupo)

        resto = resto_nuevo
        ngrupos = ngrupos_nuevo

    return puntos_capa, labels_capa, grupos_capa

def built_lista_pos(id_grupo, grupos_capa_compress, lista_pos):
    desplaz = 0
    for id in range(id_grupo):
        desplaz += grupos_capa_compress[id]
    result = lista_pos + desplaz
    return result


def search_near_centroid(id_grupo, id_centroide, id_ult_vecino, centroides_examinados, puntos_capa, labels_capa,
                         grupos_capa, ids_vecinos, vector_original, metrica):
    # 01/11: Para optimizar rendimiento, considerar cambio de pairwisedistances por cdist

    D = pairwise_distances(puntos_capa[0][id_grupo], metric=metrica)
    # min1 = D[id_centroide, 0:id_centroide].min()
    menor = np.partition(D[id_centroide], 1)[1]
    new_id_centroide = (np.argwhere(D[id_centroide] == menor)).ravel()

    if centroides_examinados[id_grupo][new_id_centroide] == 0:
        lista_pos = np.argwhere(labels_capa[0][id_grupo][:] == new_id_centroide)
        lista_pos = built_lista_pos(id_grupo, grupos_capa[0][:], lista_pos)
        lista_pos = lista_pos.ravel()

        new_lista_pos = np.setdiff1d(lista_pos, ids_vecinos)
        if len(new_lista_pos) >= 1:
            puntos_seleccionados = np.array(vector_original[new_lista_pos])
            vecino_guardado = (np.array(vector_original[id_ult_vecino])).reshape(1, 2)
            puntos_dist = np.concatenate([vecino_guardado, puntos_seleccionados])
            D = pairwise_distances(puntos_dist, metric=metrica)
            new_colum = busca_dist_menor(D)
            id_punto = new_lista_pos[new_colum - 1]
            dist = D[0, new_colum]

            if len(new_lista_pos) == 1:
                # No hay más puntos asociados con ese centroide, por lo que lo marcamos (ponemos su posición
                # a 1) como ya examinado
                centroides_examinados[id_grupo][new_id_centroide] = 1
        else:
            # Este centroide no nos vale, necesitamos otro:
            siguiente = 2
            salir = False
            while (centroides_examinados[id_grupo][new_id_centroide] == 1) and (siguiente < len(D[id_centroide]))\
                    and (not salir):
                menor = np.partition(D[id_centroide], siguiente)[siguiente]
                new_id_centroide = (np.argwhere(D[id_centroide] == menor)).ravel()

                lista_pos = np.argwhere(labels_capa[0][id_grupo][:] == new_id_centroide)
                lista_pos = built_lista_pos(id_grupo, grupos_capa[0][:], lista_pos)
                lista_pos = lista_pos.ravel()

                new_lista_pos = np.setdiff1d(lista_pos, ids_vecinos)
                if len(new_lista_pos) >= 1:
                    puntos_seleccionados = np.array(vector_original[new_lista_pos])
                    vecino_guardado = (np.array(vector_original[id_ult_vecino])).reshape(1, 2)
                    puntos_dist = np.concatenate([vecino_guardado, puntos_seleccionados])
                    D = pairwise_distances(puntos_dist, metric=metrica)
                    new_colum = busca_dist_menor(D)
                    id_punto = new_lista_pos[new_colum - 1]
                    dist = D[0, new_colum]
                    salir = True
                    if len(new_lista_pos) == 1:
                        # No hay más puntos asociados con ese centroide, por lo que lo marcamos (ponemos su posición
                        # a 1) como ya examinado
                        centroides_examinados[id_grupo][new_id_centroide] = 1
                else:
                    siguiente += 1

    else:
        # Si el centroide ha sido examinado, tengo que buscar el siguiente más cercano
        siguiente = 2
        maximo = len(D[id_centroide])
        salir = False
        while (centroides_examinados[id_grupo][new_id_centroide] == 1) and (siguiente < maximo) \
                and (not salir):
            menor = np.partition(D[id_centroide], siguiente)[siguiente]
            new_id_centroide = (np.argwhere(D[id_centroide] == menor)).ravel()

            lista_pos = np.argwhere(labels_capa[0][id_grupo][:] == new_id_centroide)
            lista_pos = built_lista_pos(id_grupo, grupos_capa[0][:], lista_pos)
            lista_pos = lista_pos.ravel()

            new_lista_pos = np.setdiff1d(lista_pos, ids_vecinos)
            if len(new_lista_pos) >= 1:
                puntos_seleccionados = np.array(vector_original[new_lista_pos])
                vecino_guardado = (np.array(vector_original[id_ult_vecino])).reshape(1, 2)
                puntos_dist = np.concatenate([vecino_guardado, puntos_seleccionados])
                D = pairwise_distances(puntos_dist, metric=metrica)
                new_colum = busca_dist_menor(D)
                id_punto = new_lista_pos[new_colum - 1]
                dist = D[0, new_colum]
                salir = True
                if len(new_lista_pos) == 1:
                    # No hay más puntos asociados con ese centroide, por lo que lo marcamos (ponemos su posición
                    # a 1) como ya examinado
                    centroides_examinados[id_grupo][new_id_centroide] = 1
            else:
                siguiente += 1


    return id_punto, dist, id_grupo, new_id_centroide

