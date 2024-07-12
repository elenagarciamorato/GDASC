from sklearn import preprocessing
from timeit import default_timer as timer
import logging
from scipy.spatial import distance
from GDASC.utils import *

# Clustering methods to be used: k-means, k-medoids
import sklearn.cluster # k-means sklearn implementation
from GDASC.clustering_algorithms import kmeans_kclust # k-means k clust implementation
import sklearn_extra.cluster  # k-medoids sklearn_extra implementation
import kmedoids as fast_kmedoids  # k-medoids fast_k-medoids (PAM) implementation


logger = logging.getLogger(__name__)

def create_tree(cant_ptos, tam_grupo, n_centroides, metrica, vector_original, dimensiones, algorithm, implementation):

    # Parámetros de entrada:
    # tam_grupo = tamaño del grupo para bombardear con los centroides (depende de la capacidad computacional).
    # n_centroides = número de centroides con los que se bombardea cada grupo
    # metric = métrica a utilizar a la hora de construir el arbol

    normaliza = False

    #    cant_ptos = nclouds * npc

    # Inicio del proceso iterativo de construcción-deconstrucción.
    start_time_constr = timer()

    vector = vector_original
    #    for iter in range(1):
    if normaliza:
        vector = preprocessing.normalize(vector, axis=0, norm='l2')

    # 23-03-2022
    #print("calculo del número de capas")
    n_capas = calculate_numcapas(cant_ptos, tam_grupo, n_centroides)

    #print("calculo de las estructuras de almacenamiento")
    puntos_capa, labels_capa, grupos_capa = built_estructuras_capa(cant_ptos, tam_grupo, n_centroides, n_capas, dimensiones)


    # Proceso iterativo para aplicar el algortimo de clustering seleccionado:
    #print("INICIO PROCESO CONSTRUCCIÓN")
    for id_capa in range(n_capas):
        #print("id_capa", id_capa)
        # Capa n:
        ngrupos = len(grupos_capa[id_capa])
        inicio = 0
        # 18-03-2021 puntos_grupo y labels_grupo ahora van a ser un np.array de tres dimensiones y los calculo
        # cuando calculo el número de grupos
        # puntos_grupo = []
        # labels_grupo = []
        cont_ptos = 0  # 03-03-2021. Contador de los puntos en cada capa
        # 23-03-2022    npuntos = []
        npuntos = np.zeros(ngrupos, dtype=int)
        for id_grupo in range(ngrupos):
            fin = inicio + tam_grupo
            # Inicio 03-03-2021. Control del último grupo (no tiene cantidad de puntos suficientes para formar
            # grupo
            if fin > cant_ptos:
                fin = cant_ptos
            # Fin 03-03-2021

            npuntos[id_grupo] = fin - inicio
            if ((fin - inicio) >= n_centroides):
                if algorithm == 'kmedoids':

                    if implementation == 'sklearnextra':

                        #precomputed_data=pairwise_distances(vector[inicio:fin], vector[inicio:fin], metric=metrica)

                        kmedoids = sklearn_extra.cluster.KMedoids(n_clusters=n_centroides, method='pam', metric=metrica).fit(vector[inicio:fin])
                        #print(kmedoids.labels_)
                        #print(kmedoids.cluster_centers_)

                        puntos_capa[id_capa][id_grupo] = kmedoids.cluster_centers_
                        labels_capa[id_capa][id_grupo] = kmedoids.labels_


                    elif implementation == 'fastkmedoids':

                        kmedoids = fast_kmedoids.KMedoids(n_clusters=n_centroides, method='fasterpam', metric=metrica).fit(vector[inicio:fin])

                        #print(kmedoids.labels_)
                        #print(kmedoids.cluster_centers_)

                        puntos_capa[id_capa][id_grupo] = kmedoids.cluster_centers_
                        labels_capa[id_capa][id_grupo] = kmedoids.labels_


                    cont_ptos += n_centroides  # 03-03-2021

                elif algorithm == 'kmeans':


                    if implementation == 'sklearn':

                        # Sklearn's Kmeans method uses as default kmeans++ to initialice centroids, Elkan's algoritm
                        # and euclidean distance (non editable)
                        print("En desuso")

                        #kmeans = KMeans(n_clusters=n_centroides, algorithm="lloyd").fit(vector[inicio:fin])

                        #puntos_capa[id_capa][id_grupo] = kmeans.cluster_centers_
                        #labels_capa[id_capa][id_grupo] = kmeans.labels_

                    elif implementation == 'kclust':

                        # THEALGORITHMS    # 10-09-2023
                        # This alternative Kmeans implementation uses sklearn.metrics.pairwise_distances(x,centroids,metric'euclidean')
                        # to associate each point with its closer centroid (this sklearn method cant be used with large
                        # datasets as GLOVE). This function can be parametrized with different distance metrics, by default
                        # it uses euclidean distance (EDITABLE ON kmeans_kclust.py). It takes as arg the initial centroids Unlike it does kmeans++
                        # that can be generated randomly using a function provided on the same script or in a customized way
                        # (we can choose to generate them using kmeans++ but WARNING it may use euclidean distance as default)

                        # Generate initial centers using kmeans++
                        # initial_centers = kmeans_plusplus_initializer(vector[inicio:fin], n_centroides).initialize()

                        # Generate initial centers using function provided by k_means_clust
                        initial_centers_kclust = kmeans_kclust.get_initial_centroids(data=vector[inicio:fin], k=n_centroides).tolist()
                        initial_centers = list([np.array(x) for x in initial_centers_kclust])

                        # Generate centroids and clusters using kmeans implementation provided by k_means_clust
                        kmeans = kmeans_kclust.kmeans(data=vector[inicio:fin], k=n_centroides, initial_centroids=initial_centers, metric=metrica)

                        #if id_grupo==1:

                            #print(kmeans[0])
                            #print(kmeans[1])

                        puntos_capa[id_capa][id_grupo] = kmeans[0]
                        labels_capa[id_capa][id_grupo] = kmeans[1]


                    cont_ptos += n_centroides  # 03-03-2021


                else:   # En principio, nunca se accede

                    print("Es necesario añadir un algoritmo de clustering valido")

            else:
                # Si los puntos que tenemos en el grupo no es mayor que el número de centroides/medoides, no hacemos culster
                # 03-03-2021  puntos_grupo.append(vector[inicio:fin])  # aquí tenemos almacenados los puntos de la
                # siguiente capa para cada grupo
                # 18-03-2021 puntos_grupo.append(np.array(vector[inicio:fin]))  # aquí tenemos almacenados los puntos de la
                # 23-03-2022 puntos_grupo[id_grupo] = np.array(vector[inicio:fin])
                puntos_capa[id_capa][id_grupo] = np.array(vector[inicio:fin])
                # siguiente capa para cada grupo
                cont_ptos = cont_ptos + (fin - inicio)  # 03-03-2021
                etiquetas = []
                for i in range((fin - inicio)):
                    etiquetas.append(i)
                # 18-03-2021 labels_grupo.append(np.array(etiquetas))
                # 23-03-2022 labels_grupo[id_grupo] = np.array(etiquetas)
                labels_capa[id_capa][id_grupo] = np.array(etiquetas)

            inicio = fin

        # 23-03-2022    grupos_capa.append(npuntos)
        grupos_capa[id_capa] = npuntos

        # Guardamos los centroides de la capa para poder hacer el proceso inverso
        vector = puntos_capa[id_capa]
        vector = np.concatenate(vector).ravel().tolist()  # 03-03-2021
        vector = np.array(vector)
        vector = vector.reshape(cont_ptos, dimensiones)

        # Calculamos el numero de grupos de la siguiente capa
        cant_ptos = cont_ptos  # 03-03-2021 Actualizamos cant_ptos con el número de puntos del siguiente nivel
        # id_capa += 1

    #print("FIN PROCESO CONSTRUCCIÓN")

    # 23-03-2022    n_capas = id_capa - 1
    end_time_constr = timer()
    # print("--- %s seconds ---", end_time_constr-start_time_constr)
    logger.info('tree time=%s seconds', end_time_constr - start_time_constr)

    return n_capas, grupos_capa, puntos_capa, labels_capa



def knn_search(n_capas, n_centroides, seq_buscada, vector_original, vecinos, centroides_examinados,
                  n, metrica, grupos_capa, puntos_capa, labels_capa):

    # En principio, esta búsqueda sobre el árbol seria exactamente igual que con MASK (mask_search)
    print("********************PROCESO DECONSTRUCCIÓN*********************")
    # start_time_deconstr = timer()
#   n_capas = id_capa - 1
    logger.info('tree-depth=%s', n_capas)

    # lcorrespond = []
    # aciertos = 0
    # fallos = 0
    # vector_aux = []
    # vector_aux = np.empty((len(vector_original), 2), float)
    lista_pos = np.empty(100, int)
    # for i in range(len(vector_original)):
    # print('buscando punto ', i)
    # seq_buscada = np.array(vector_original[i])
    # seq_buscada = np.reshape(seq_buscada, (1, 2))

    seq_buscada = np.reshape(seq_buscada, (1, 2))
    for id_capa in range(n_capas-1, -1, -1):
        # 03-03-2021 Obtenemos los centroides de la capa
        centroides = puntos_capa[id_capa]
        centroides = np.concatenate(centroides) #.ravel() #.tolist()

        # 23-03-2022    if id_capa < n_capas:
        if id_capa < (n_capas - 1):
            # seleccionamos solo los puntos que están asociados con ese centroide
            centroides = np.array(centroides[lista_pos])

        puntos_dist = np.concatenate([seq_buscada, centroides])
        D = pairwise_distances(puntos_dist, metric=metrica)     # euclidean, chebyshev, manhattan
        columna = busca_dist_menor(D)
        # Corrección del índice del centroide
        # 23-03-2022    if id_capa != n_capas:
        if id_capa != (n_capas - 1):
            pos_centroide = lista_pos[columna - 1]
            if pos_centroide >= n_centroides:
                id_grupo = int(pos_centroide / n_centroides)
                id_centroide = pos_centroide - (id_grupo * n_centroides)
            else:
                id_centroide = pos_centroide
                id_grupo = 0
        else:
            # 08-03-2021. Corrección para cuando la última capa del arbol tiene más de un grupo
            if len(grupos_capa[id_capa]) > 1:
                if (columna - 1) >= n_centroides:
                    id_grupo = int((columna - 1) / n_centroides)
                    id_centroide = (columna - 1) - (id_grupo * n_centroides)
                else:
                    id_centroide = columna - 1
                    id_grupo = 0
                # 08-03-2021. Fin.
            else:
                id_centroide = columna - 1
                id_grupo = 0

        lista_pos_aux = np.argwhere(labels_capa[id_capa][id_grupo][:] == id_centroide)
        lista_pos = built_lista_pos(id_grupo, grupos_capa[id_capa][:], lista_pos_aux)
        lista_pos = lista_pos.ravel()

    # Capa de los datos:
    puntos_seleccionados = np.array(vector_original[lista_pos])
    puntos_dist = np.concatenate([seq_buscada, puntos_seleccionados])
    # D = pairwise_distances(puntos_dist, metric=metrica)
    D = distance.pdist(puntos_dist, metric=metrica)  # scipy resulta más eficiente
    columna = busca_dist_menor(D)
    id_punto = lista_pos[columna - 1]

    # Control de los vecinos guardados (ciclados)
    # Si el id_punto encontrado ya lo teniamos guardado en vecinos, nos quedamos con el siguiente
    # mas cercano
    vecino = np.empty(5, object)
    almacenado = False
    if n == 0:
        # Guardamos directamente el vecino encontrado (es el primero)
        vecino[0] = id_punto
        vecino[1] = D[0, columna]
        vecino[2] = vector_original[id_punto]
        vecino[3] = id_grupo
        vecino[4] = id_centroide
        vecinos[n] = vecino
        almacenado = True
        if len(lista_pos) == 1:
            # No hay más puntos asociados con ese centroide, por lo que lo marcamos (ponemos su posición
            # a 1) como ya examinado
            centroides_examinados[id_grupo][id_centroide] = 1
    else:
        # Buscamos si el nuevo vecino esta ya guardado
        ids_vecinos = np.zeros(n, dtype=int)
        for x in range(n):
            ids_vecinos[x] = vecinos[x][0]
        index = np.ravel(np.asarray(ids_vecinos == id_punto).nonzero())

        if len(index) == 0:
            # No lo tenemos guardado, por lo tanto, lo guardamos
            vecino[0] = id_punto
            vecino[1] = D[0, columna]
            vecino[2] = vector_original[id_punto]
            vecino[3] = id_grupo
            vecino[4] = id_centroide
            vecinos[n] = vecino
            almacenado = True
            if len(lista_pos) == 1:
                # No hay más puntos asociados con ese centroide, por lo que lo marcamos (ponemos su posición
                # a 1) como ya examinado
                centroides_examinados[id_grupo][id_centroide] = 1
        else:
            # Si lo tenemos guardado. Buscamos otro candidato
            if len(lista_pos) == 1:
                # No tengo más candidatos asociados a ese centriode. Hay que buscar un nuevo centroide y examinar
                # sus candidatos
                id_ult_vecino = vecinos[n-1][0]
                id_punto, dist, id_grupo, new_id_centroide = search_near_centroid(id_grupo, id_centroide,
                                                            id_ult_vecino, centroides_examinados, puntos_capa,
                                                            labels_capa, grupos_capa, ids_vecinos,
                                                            vector_original, metrica)
                vecino[0] = id_punto
                vecino[1] = dist
                vecino[2] = vector_original[id_punto]
                vecino[3] = id_grupo
                vecino[4] = new_id_centroide
                vecinos[n] = vecino
                almacenado = True
            else:
                # Tenemos más candidatos asociados a ese centroide. Buscamos el siguiente punto más cercano
                new_lista_pos = np.setdiff1d(lista_pos, ids_vecinos)
                if len(new_lista_pos) == 0:
                    id_ult_vecino = vecinos[n - 1][0]
                    id_punto, dist, id_grupo, new_id_centroide = search_near_centroid(id_grupo, id_centroide,
                                                                                      id_ult_vecino,
                                                                                      centroides_examinados,
                                                                                      puntos_capa, labels_capa,
                                                                                      grupos_capa, ids_vecinos,
                                                                                      vector_original, metrica)
                    vecino[0] = id_punto
                    vecino[1] = dist
                    vecino[2] = vector_original[id_punto]
                    vecino[3] = id_grupo
                    vecino[4] = new_id_centroide
                    vecinos[n] = vecino
                    almacenado = True
                else:
                    puntos_seleccionados = np.array(vector_original[new_lista_pos])
                    vecino_guardado = (np.array(vector_original[id_punto])).reshape(1,2)
                    puntos_dist = np.concatenate([vecino_guardado, puntos_seleccionados])
                    D = pairwise_distances(puntos_dist, metric=metrica)
                    new_colum = busca_dist_menor(D)
                    id_punto = new_lista_pos[new_colum - 1]

                    vecino[0] = id_punto
                    vecino[1] = D[0, new_colum]
                    vecino[2] = vector_original[id_punto]
                    vecino[3] = id_grupo
                    vecino[4] = id_centroide
                    vecinos[n] = vecino
                    almacenado = True
                    if len(new_lista_pos) == 1:
                        # No hay más puntos asociados con ese centroide, por lo que lo marcamos (ponemos su posición
                        # a 1) como ya examinado
                        centroides_examinados[id_grupo][id_centroide] = 1

    print("FIN PROCESO DECONSTRUCCIÓN")
    # end_time_deconstr = timer()
    # print("--- %s seconds ---", end_time_deconstr-start_time_deconstr)
    # logger.info('search time= %s seconds', end_time_deconstr - start_time_deconstr)

    return almacenado


def knn_approximate_search(n_centroides, punto_buscado, vector_original, k_vecinos, metrica,
                         grupos_capa, puntos_capa, labels_capa, dimensiones, radio):

    # ACTUALIZACION 30/09/23 - A la hora de buscar los mas vecinos más cercanos, se utiliza tambien la métrica
    # que se pasa como argumento y que teoricamente debe ser la misma con la que se construyó el arbol,
    # calculando las distancias a traves de la función de scipy cdist(punto, puntos, metrica)

    # La búsqueda es aproximada porque se limita a un radio

    if metrica == 'manhattan':  metrica = 'cityblock'  # SCIPY - cdist (necesario traducir manhattan como cityblock)

    #print("********************PROCESO DECONSTRUCCIÓN*********************")
    # logger.info('tree-depth=%s', n_capas)

    # lista_pos = np.empty(100, int)
    # lista_ptos = []
    # aux_vecinos = np.empty(1, object)

    punto_buscado = np.reshape(punto_buscado, (1, dimensiones))

    # Voy directamente a la capa 1 (la que está encima de los datos), establecemos como radio 3 veces la
    # distancia menor
    id_capa = 0
    centroides = puntos_capa[id_capa]
    centroides = np.concatenate(centroides)
    puntos_dist = np.concatenate([punto_buscado, centroides])

    D = distance.cdist(punto_buscado, centroides, metric=metrica)[0]
    # print(punto_buscado)
    # print (centroides.shape)
    # print(centroides[0])
    # print(D)
    # dist_nearest_centroid = np.partition(D, 1)[1]
    radius = radio  # 3 * dist_nearest_centroid (1.15 glove completo, 3 glove100000, 5 MNIST)

    # Para cada uno de los centroides con distancia menor a radius nos quedamos con k vecinos más cercanos
    # Primero almacenamos los índices de los centroides que cumplen la condición
    # filad = D[0, 1:]
    filad = D
    selec_centroides = np.array(np.flatnonzero(filad<=radius)) #+ 1
    # print("Centroides seleccionados: " + str(selec_centroides.shape))
    # coords_centroides = centroides[selec_centroides]
    ids_selec_centroides = np.empty(len(selec_centroides), tuple)
    tam = 0
    for i in range(len(selec_centroides)):
        # Corrección del indice del centroide
        if selec_centroides[i] >= n_centroides:
            id_grupo = int(selec_centroides[i] / n_centroides)
            id_centroide = selec_centroides[i] - (id_grupo * n_centroides)
        else:
            id_centroide = selec_centroides[i]
            id_grupo = 0
        ids_selec_centroides[i] = (id_grupo, id_centroide)
        tam += np.count_nonzero(labels_capa[id_capa][id_grupo][:] == id_centroide)

    ids_selec_points = np.empty(tam, int)
    ini = 0
    fin = 0
    for idg,idc in ids_selec_centroides:
        lista_pos_aux = np.argwhere(labels_capa[id_capa][idg][:] == idc)
        lista_pos = built_lista_pos(idg, grupos_capa[id_capa][:], lista_pos_aux)
        lista_pos = np.reshape(lista_pos, len(lista_pos))
        fin += len(lista_pos)
        ids_selec_points[ini:fin] = lista_pos
        ini = fin

    # De todos los puntos seleccionados solo guardamos los que cumplen la condición de la distancia (puntos_cercanos)
    puntos_seleccionados = np.array(vector_original[ids_selec_points])
    #print("Puntos seleccionados: " + str(puntos_seleccionados.shape))
    #print("Punto buscado: " + str(punto_buscado.shape))
    #print("Distancia entre el punto buscado y los puntos seleccionados:")
    dist = distance.cdist(np.array(punto_buscado), np.array(puntos_seleccionados), metric=metrica)

    aux_ids_points = np.array(np.nonzero(dist<=radius))    # +1
    aux_ids_points = aux_ids_points[1]
    ids_points = ids_selec_points[aux_ids_points]
    dist_points = dist[dist <= radius]

    #print("Puntos dentro del rango de distancia: " + str(dist_points.size))
    #print(dist_points)

    puntos_cercanos = np.empty((len(ids_points),3), object)
    for i in range(len(ids_points)):
        puntos_cercanos[i][0] = ids_points[i]
        puntos_cercanos[i][1] = dist_points[i]
        puntos_cercanos[i][2] = vector_original[ids_points[i]]


    #print("FIN PROCESO DECONSTRUCCIÓN\n")
    # end_time_deconstr = timer()
    # print("--- %s seconds ---", end_time_deconstr-start_time_deconstr)
    # logger.info('search time= %s seconds', end_time_deconstr - start_time_deconstr)

    # Creamnos las estructuras para almacenar los datos relativos a los vecinos
    indices_k_vecinos = np.empty(k_vecinos, dtype=int)
    coords_k_vecinos = np.empty([k_vecinos, vector_original.shape[1]], dtype=float)
    dists_k_vecinos = np.empty(k_vecinos, dtype=float)

    # Completar el array de puntos cercanos  con None s hasta llegar al tamaño de vecinos deseado (k_vecinos)
    # Esto evita el error index out of bounds
    if len(puntos_cercanos) < k_vecinos:
        puntos_cercanos= np.append(puntos_cercanos, np.full((k_vecinos - len(puntos_cercanos), 3), None), axis=0)

    # Ordenamos los puntos en base a su distancia con el punto de query
    idx = np.argsort(puntos_cercanos[:, 1])

    # Designamos los k_vecinos puntos con menor distancia al punto de consulta como vecinos
    for i in range(k_vecinos):
        indices_k_vecinos[i] = puntos_cercanos[idx[i]][0]
        coords_k_vecinos[i, :] = puntos_cercanos[idx[i]][2]
        dists_k_vecinos[i] = puntos_cercanos[idx[i]][1]

    return [indices_k_vecinos, coords_k_vecinos, dists_k_vecinos]