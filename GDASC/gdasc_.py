import numpy as np
from sklearn import preprocessing
from timeit import default_timer as timer
import logging
from scipy.spatial import distance
from GDASC.GDASC.utils import *

# Clustering methods to be used: k-means, k-medoids
import sklearn.cluster  # k-means sklearn implementation
from GDASC.GDASC.clustering_algorithms import kmeans_kclust  # k-means k clust implementation
import sklearn_extra.cluster  # k-medoids sklearn_extra implementation
import kmedoids as fast_kmedoids  # k-medoids fast_k-medoids (PAM) implementation

logger = logging.getLogger(__name__)


def create_tree(cant_ptos, tam_grupo, n_centroides, metrica, vector_original, dimensiones, algorithm, implementation):
    """
    Constructs a hierarchical tree structure using clustering algorithms.

    Parameters:
    cant_ptos : int
            Total number of points.
    tam_grupo : int
            Size of the group to be clustered with centroids.
    n_centroides : int
            Number of centroids to be used in each clustering.
    metrica   : str
            Metric to be used for distance calculation in clustering.
    vector_original : np.array
            Original data points.
    dimensiones   : int
            Dimensionality of the data points.
    algorithm : str
            Clustering algorithm to be used ('kmedoids' or 'kmeans').
    implementation : str
            Specific implementation of the clustering algorithm to use.

    Returns:
    tuple: A tuple containing:
        - n_capas  : int
                Number of layers in the hierarchical tree.
        - grupos_capa : list
                Group structure for each layer.
        - puntos_capa : list
                Centroids for each group in each layer.
        - labels_capa : list
                Labels assigned to each point in each group for each layer.
    """

    normaliza = False

    # Inicio del proceso iterativo de construcción-deconstrucción.
    start_time_constr = timer()

    vector = vector_original
    if normaliza:
        vector = preprocessing.normalize(vector, axis=0, norm='l2')

    n_capas = calculate_numcapas(cant_ptos, tam_grupo, n_centroides)
    puntos_capa, labels_capa, grupos_capa = built_estructuras_capa(cant_ptos, tam_grupo, n_centroides, n_capas,
                                                                   dimensiones)

    # Proceso iterativo para aplicar el algortimo de clustering seleccionado:
    for id_capa in range(n_capas):
        ngrupos = len(grupos_capa[id_capa])
        inicio = 0
        # puntos_grupo y labels_grupo ahora van a ser un np.array de tres dimensiones y los calculo
        # cuando calculo el número de grupos
        cont_ptos = 0  # Contador de los puntos en cada capa
        npuntos = np.zeros(ngrupos, dtype=int)
        for id_grupo in range(ngrupos):
            fin = inicio + tam_grupo
            # Control del último grupo (no tiene cantidad de puntos suficientes para formar
            # grupo
            if fin > cant_ptos:
                fin = cant_ptos
            npuntos[id_grupo] = fin - inicio
            if ((fin - inicio) >= n_centroides):
                if algorithm == 'kmedoids':

                    if implementation == 'sklearnextra':

                        kmedoids = sklearn_extra.cluster.KMedoids(n_clusters=n_centroides, method='pam',
                                                                  metric=metrica).fit(vector[inicio:fin])
                        puntos_capa[id_capa][id_grupo] = kmedoids.cluster_centers_
                        labels_capa[id_capa][id_grupo] = kmedoids.labels_


                    elif implementation == 'fastkmedoids':

                        kmedoids = fast_kmedoids.KMedoids(n_clusters=n_centroides, method='fasterpam',
                                                          metric=metrica).fit(vector[inicio:fin])
                        puntos_capa[id_capa][id_grupo] = kmedoids.cluster_centers_
                        labels_capa[id_capa][id_grupo] = kmedoids.labels_

                    cont_ptos += n_centroides

                elif algorithm == 'kmeans':

                    if implementation == 'sklearn':

                        # Sklearn's Kmeans method uses as default kmeans++ to initialice centroids, Elkan's algoritm
                        # and euclidean distance (non editable)
                        print("En desuso")

                    elif implementation == 'kclust':

                        # THE ALGORITHMS
                        # This alternative Kmeans implementation uses sklearn.metrics.pairwise_distances(x,centroids,metric'euclidean')
                        # to associate each point with its closer centroid (this sklearn method cant be used with large
                        # datasets as GLOVE). This function can be parametrized with different distance metrics, by default
                        # it uses euclidean distance (EDITABLE ON kmeans_kclust.py). It takes as arg the initial centroids Unlike it does kmeans++
                        # that can be generated randomly using a function provided on the same script or in a customized way
                        # (we can choose to generate them using kmeans++ but WARNING it may use euclidean distance as default)

                        # Generate initial centers using function provided by k_means_clust
                        initial_centers_kclust = kmeans_kclust.get_initial_centroids(data=vector[inicio:fin],
                                                                                     k=n_centroides).tolist()
                        initial_centers = list([np.array(x) for x in initial_centers_kclust])

                        # Generate centroids and clusters using kmeans implementation provided by k_means_clust
                        kmeans = kmeans_kclust.kmeans(data=vector[inicio:fin], k=n_centroides,
                                                      initial_centroids=initial_centers, metric=metrica)
                        puntos_capa[id_capa][id_grupo] = kmeans[0]
                        labels_capa[id_capa][id_grupo] = kmeans[1]

                    cont_ptos += n_centroides


                else:  # En principio, nunca se accede

                    print("Es necesario añadir un algoritmo de clustering valido")

            else:

                puntos_capa[id_capa][id_grupo] = np.array(vector[inicio:fin])
                # siguiente capa para cada grupo
                cont_ptos = cont_ptos + (fin - inicio)
                etiquetas = []
                for i in range((fin - inicio)):
                    etiquetas.append(i)

                labels_capa[id_capa][id_grupo] = np.array(etiquetas)

            inicio = fin

        grupos_capa[id_capa] = npuntos

        # Guardamos los centroides de la capa para poder hacer el proceso inverso
        vector = puntos_capa[id_capa]
        vector = np.concatenate(vector).ravel().tolist()
        vector = np.array(vector)
        vector = vector.reshape(cont_ptos, dimensiones)

        # Calculamos el numero de grupos de la siguiente capa
        cant_ptos = cont_ptos  # Actualizamos cant_ptos con el número de puntos del siguiente nivel

    end_time_constr = timer()

    logger.info('tree time=%s seconds', end_time_constr - start_time_constr)

    return n_capas, grupos_capa, puntos_capa, labels_capa


def knn_search(n_capas, n_centroides, seq_buscada, vector_original, vecinos, centroides_examinados,
               n, metrica, grupos_capa, puntos_capa, labels_capa):
    """
    Performs a k-nearest neighbors (KNN) search using a hierarchical tree structure.

    Parameters:
    n_capas         : int
            Number of layers in the hierarchical tree.
    n_centroides          : int
            Number of centroids in each group.
    seq_buscada           : numpy.ndarray
            The query point for which the nearest neighbors are to be found.
    vector_original       : numpy.ndarray
            The original dataset of points.
    vecinos               : list
            List to store the nearest neighbors found during the search.
    centroides_examinados : numpy.ndarray
            Array to track which centroids have been examined.
    n : int
            Current number of nearest neighbors found.
    metrica               : str
            The distance metric to use for calculating distances.
    grupos_capa           : list
            Number of points in each group at each layer.
    puntos_capa           : list
            Cluster centroids at each layer.
    labels_capa           : list
            Labels assigned to each point at each layer.

    Returns:
    bool                   : True if a new neighbor was stored, False otherwise.
    """

    # En principio, esta búsqueda sobre el árbol seria exactamente igual que con MASK (mask_search)
    print("******************** DECONSTRUCTION PROCESS *********************")
    logger.info('tree-depth=%s', n_capas)
    lista_pos = np.empty(100, int)
    # Reshape the query sequence
    seq_buscada = np.reshape(seq_buscada, (1, 2))
    # Iterate over each layer from the deepest to the root
    for id_capa in range(n_capas - 1, -1, -1):
        # Obtain centroids of the current layer
        centroides = puntos_capa[id_capa]
        centroides = np.concatenate(centroides)

        # Select points associated with the current centroid if not at the deepest layer
        if id_capa < (n_capas - 1):
            # seleccionamos solo los puntos que están asociados con ese centroide
            centroides = np.array(centroides[lista_pos])

        puntos_dist = np.concatenate([seq_buscada, centroides])
        D = pairwise_distances(puntos_dist, metric=metrica)  # euclidean, chebyshev, manhattan
        columna = busca_dist_menor(D)
        # Correct centroid index
        if id_capa != (n_capas - 1):
            pos_centroide = lista_pos[columna - 1]
            if pos_centroide >= n_centroides:
                id_grupo = int(pos_centroide / n_centroides)
                id_centroide = pos_centroide - (id_grupo * n_centroides)
            else:
                id_centroide = pos_centroide
                id_grupo = 0
        else:
            # Corrección para cuando la última capa del arbol tiene más de un grupo
            if len(grupos_capa[id_capa]) > 1:
                if (columna - 1) >= n_centroides:
                    id_grupo = int((columna - 1) / n_centroides)
                    id_centroide = (columna - 1) - (id_grupo * n_centroides)
                else:
                    id_centroide = columna - 1
                    id_grupo = 0
            else:
                id_centroide = columna - 1
                id_grupo = 0

        lista_pos_aux = np.argwhere(labels_capa[id_capa][id_grupo][:] == id_centroide)
        lista_pos = built_lista_pos(id_grupo, grupos_capa[id_capa][:], lista_pos_aux)
        lista_pos = lista_pos.ravel()

    # Select points at the data layer
    puntos_seleccionados = np.array(vector_original[lista_pos])
    puntos_dist = np.concatenate([seq_buscada, puntos_seleccionados])
    D = distance.pdist(puntos_dist, metric=metrica)  # scipy resulta más eficiente
    columna = busca_dist_menor(D)
    id_punto = lista_pos[columna - 1]

    # Control de los vecinos guardados (ciclados)
    # Si el id_punto encontrado ya lo teniamos guardado en vecinos, nos quedamos con el siguiente
    # mas cercano
    # Store the found neighbor or find another candidate
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
                id_ult_vecino = vecinos[n - 1][0]
                id_punto, dist, id_grupo, new_id_centroide = search_near_centroid(id_grupo, id_centroide,
                                                                                  id_ult_vecino, centroides_examinados,
                                                                                  puntos_capa,
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
                    vecino_guardado = (np.array(vector_original[id_punto])).reshape(1, 2)
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

    print("END OF DECONSTRUCTION PROCESS")

    return almacenado


def knn_approximate_search(n_centroides, punto_buscado, vector_original, k_vecinos, metrica,
                           grupos_capa, puntos_capa, labels_capa, dimensiones, radio):
    """
    Performs an approximate k-nearest neighbors (KNN) search using a hierarchical tree structure
    and a search radius.

    Parameters:
    n_centroides : int
            Number of centroids in each group.
    punto_buscado : numpy.ndarray
                The query point for which the nearest neighbors are to be found.
    vector_original : numpy.ndarray
                The original dataset of points.
    k_vecinos : int
            Number of nearest neighbors to find.
    metrica : str
            The distance metric to use for calculating distances.
    grupos_capa :  list
             Number of points in each group at each layer.
    puntos_capa : list
            Cluster centroids at each layer.
    labels_capa : list
             Labels assigned to each point at each layer.
    dimensiones : int
            Number of dimensions of the points.
    radio : float
        Search radius for the approximate search.

    Returns:
    list: A list containing three elements:
        - numpy.ndarray: Indices of the k nearest neighbors.
        - numpy.ndarray: Coordinates of the k nearest neighbors.
        - numpy.ndarray: Distances to the k nearest neighbors.
    """
    #  A la hora de buscar los mas vecinos más cercanos, se utiliza tambien la métrica
    # que se pasa como argumento y que teoricamente debe ser la misma con la que se construyó el arbol,
    # calculando las distancias a traves de la función de scipy cdist(punto, puntos, metrica)

    # La búsqueda es aproximada porque se limita a un radio

    # Update the metric name for compatibility with scipy
    if metrica == 'manhattan':
        metrica = 'cityblock'  # scipy cdist requires 'cityblock' instead of 'manhattan'

    punto_buscado = np.reshape(punto_buscado, (1, dimensiones))
    # Directly move to layer 1 (the one above the data layer) and set the search radius( we set as radius 3 times the smallest distance)

    id_capa = 0
    centroides = puntos_capa[id_capa]
    centroides = np.concatenate(centroides)
    puntos_dist = np.concatenate([punto_buscado, centroides])

    D = distance.cdist(punto_buscado, centroides, metric=metrica)[0]

    radius = radio  # 3 * dist_nearest_centroid (1.15 glove completo, 3 glove100000, 5 MNIST)

    # Para cada uno de los centroides con distancia menor a radius nos quedamos con k vecinos más cercanos
    # Primero almacenamos los índices de los centroides que cumplen la condición
    # filad = D[0, 1:]

    # Find centroids within the search radius
    selec_centroides = np.flatnonzero(D <= radius)
    # Preallocate the array to store centroid group and ID for each of them
    ids_selec_centroides = np.empty(len(selec_centroides), dtype=object)

    # 29/09/24 CODE IMPROVEMENT: Obtain group and centroid IDs in a vectorized manner

    for i, sc in enumerate(selec_centroides):
        id_grupo = sc // n_centroides  # Integer division to get the group ID
        id_centroide = sc % n_centroides  # Modulo operation to get the centroid ID
        ids_selec_centroides[i] = (id_grupo, id_centroide)

    # Compute the total size of selected centroids based on labels
    tam = sum(np.count_nonzero(labels_capa[id_capa][id_grupo][:] == id_centroide)
              for id_grupo, id_centroide in ids_selec_centroides)

    # Find the points associated with those centroids
    ids_selec_points = np.empty(tam, int)
    ini = 0
    fin = 0
    for idg, idc in ids_selec_centroides:
        lista_pos_aux = np.argwhere(labels_capa[id_capa][idg][:] == idc)
        lista_pos = built_lista_pos(idg, grupos_capa[id_capa][:], lista_pos_aux)
        lista_pos = np.reshape(lista_pos, len(lista_pos))
        fin += len(lista_pos)
        ids_selec_points[ini:fin] = lista_pos
        ini = fin

    # Within all the selected points, we would only store those who meet the condition (dist<radius)
    puntos_seleccionados = np.array(vector_original[ids_selec_points])
    dist = distance.cdist(np.array(punto_buscado), np.array(puntos_seleccionados), metric=metrica)

    aux_ids_points = np.array(np.nonzero(dist <= radius))  # +1
    aux_ids_points = aux_ids_points[1]
    ids_points = ids_selec_points[aux_ids_points]
    dist_points = dist[dist <= radius]

    puntos_cercanos = np.empty((len(ids_points), 3), object)
    for i in range(len(ids_points)):
        puntos_cercanos[i][0] = ids_points[i]
        puntos_cercanos[i][1] = dist_points[i]
        puntos_cercanos[i][2] = vector_original[ids_points[i]]

    # Creamos las estructuras para almacenar los datos relativos a los vecinos
    # Structures to store the nearest neighbors
    indices_k_vecinos = np.empty(k_vecinos, dtype=int)
    coords_k_vecinos = np.empty([k_vecinos, vector_original.shape[1]], dtype=float)
    dists_k_vecinos = np.empty(k_vecinos, dtype=float)

    # Completar el array de puntos cercanos  con None s hasta llegar al tamaño de vecinos deseado (k_vecinos)
    # Esto evita el error index out of bounds

    # Pad the array of close points with None until it reaches the size of k neighbors
    if len(puntos_cercanos) < k_vecinos:
        puntos_cercanos = np.append(puntos_cercanos, np.full((k_vecinos - len(puntos_cercanos), 3), None), axis=0)

    # Sort points by distance to the query point
    # Ordenamos los puntos en base a su distancia con el punto de query
    idx = np.argsort(puntos_cercanos[:, 1])

    # Select the k closest points as neighbors
    # Designamos los k_vecinos puntos con menor distancia al punto de consulta como vecinos
    for i in range(k_vecinos):
        indices_k_vecinos[i] = puntos_cercanos[idx[i]][0]
        coords_k_vecinos[i, :] = puntos_cercanos[idx[i]][2]
        dists_k_vecinos[i] = puntos_cercanos[idx[i]][1]

    return [indices_k_vecinos, coords_k_vecinos, dists_k_vecinos]


def find_centroid_group(inheritage, grupos_capa, n_centroids, subgroup):
    """
    Determines the group of a centroid in a hierarchical tree structure.

    Parameters:
    inheritage : list
        List of ancestor groups leading to the current group.
    grupos_capa : list
        Number of points in each group at each layer.
    n_centroids : int
        Number of centroids in each group.
    subgroup : int
        Subgroup index within the current group.

    Returns:
    int
        The group index of the centroid.
    """

    tg = grupos_capa[-1][0]
    n_branches = tg//n_centroids
    father_group = inheritage[-1]
    centroid_group = father_group * n_branches + subgroup

    return centroid_group


def explore_centroid(punto_buscado, current_layer, inheritage, current_centroid_id, coords_puntos_capas, puntos_capas, grupos_capa, n_centroides, metrica, radio, neighbours):
    """
        Explores the hierarchical tree structure to find centroids within a given radius.

        Parameters:
        punto_buscado : numpy.ndarray
            The query point for which the nearest neighbors are to be found.
        current_layer : int
            The current layer in the hierarchical tree being explored.
        inheritage : list
            List of ancestor groups leading to the current group.
        current_centroid_id : int
            The ID of the current centroid being explored.
        coords_puntos_capas : list
            Coordinates of points at each layer.
        puntos_capas : list
            Cluster centroids at each layer.
        grupos_capa : list
            Number of points in each group at each layer.
        n_centroides : int
            Number of centroids in each group.
        metrica : str
            The distance metric to use for calculating distances.
        radio : float
            Search radius for the approximate search.
        neighbours : list
            List to store the nearest neighbors found during the search.

        Returns:
        list
            Updated list of nearest neighbors.
        """

    #print("Yo soy el centroide " + str(current_centroid_id) + " con coordenadas " + str(current_centroid_coords))
    # print("Yo soy el centroide " + str(current_centroid_id) + " con distancia al punto q " + str(centroid_distance_q))
    #current_group = current_centroid_id // n_centroides  # Integer division to get the group ID
    #id_centroide = current_centroid_id % n_centroides  # Modulo operation to get the centroid ID

    if current_layer == grupos_capa.size:
        # In the first layer, the number of groups is 1
        groups_current_layer = 1
    else:
        # In the others, it depends on the index structure
        groups_current_layer = grupos_capa[current_layer].size

    # The group to which the centroid belongs is known based on where it comes from
    prototype_group = inheritage[-1]

    print("Exploro el prototipo ", current_centroid_id, " perteneciente a la capa ", current_layer, " y el grupo ", inheritage[-1], " de un total de ", groups_current_layer, " grupos")

    # I take the points on the next layer that are associated with the current centroid
    id_prototypes_layer_down = puntos_capas[current_layer-1][prototype_group] # contiene 70 elem en la primera it
    #print(id_prototypes_layer_down)

    id_associated_prototypes_layer_down = np.where(id_prototypes_layer_down == current_centroid_id)[0]
    #print(id_associated_prototypes_layer_down)

    associated_prototypes_layer_down = []  # Empty array containing groups_down subarrays

    for i in range(0, len(id_associated_prototypes_layer_down)):
        # Obtain the subgroup to which the prototype belongs
        subgroup = id_associated_prototypes_layer_down[i] // n_centroides
        group = find_centroid_group(inheritage, grupos_capa, n_centroides, subgroup)
        print("Mapeo al hijo con indice ", (id_associated_prototypes_layer_down[i]), " perteneciente al grupo ", group, " de la capa ", current_layer-1)

        # Correct the index of the prototype
        id_associated_prototypes_layer_down[i] = id_associated_prototypes_layer_down[i] % n_centroides
        #print("El hijo " + str(id_associated_prototypes_layer_down[i]) + " pertenece al grupo " + str(id_grupo) + " -> Indice corregido")


        prototype_layer_down = np.empty(4, dtype=object)
        prototype_layer_down[0] = id_associated_prototypes_layer_down[i]
        prototype_layer_down[1] = group

        if current_layer-1 == 0:
            prototype_layer_down[2] = None

        else:
            prototype_layer_down[2] = coords_puntos_capas[current_layer-2][prototype_layer_down[1]][prototype_layer_down[0]]

        associated_prototypes_layer_down.append(prototype_layer_down)


    associated_prototypes_layer_down = np.array(associated_prototypes_layer_down)

    #print("Los puntos mapeados por este centroide son: " + str(grouped_associated_prototypes_layer_down[:,1]))  # Hasta aqui benne!!!

    if current_layer == 1:
        # We would explore all the points associated to this centroid
        for i in range(0, len(associated_prototypes_layer_down)):
            neighbour_id = n_centroides*associated_prototypes_layer_down[i][1]+associated_prototypes_layer_down[i][0]
            neighbours.append(neighbour_id)

        return neighbours

    else:

        # We would only explore those who meet the condition / are within a radius (dist<radius)
        #coordinates_bottomed_prototypes = np.reshape(grouped_associated_prototypes_layer_down[:, 2][0], (1, 30))
        coordinates_bottomed_prototypes = np.vstack(associated_prototypes_layer_down[:, 2])
        distances_bottomed_prototypes = distance.cdist(np.array(punto_buscado), coordinates_bottomed_prototypes, metric=metrica)
        associated_prototypes_layer_down[:, 3] = distances_bottomed_prototypes
        print("Las distancias de los hijos son: " + str(distances_bottomed_prototypes))  # Se coge la primera dimension del array porque por el comportamiendo de cdist se necesita array bidimensional, pero resulta redundante

        # We would only explore those who meet the condition / are within a radius (dist<radius)
        explorable_prototypes_indices = np.where(associated_prototypes_layer_down[:, 3] <= radio)[0]
        explorable_prototypes = associated_prototypes_layer_down[explorable_prototypes_indices]
        print("Los siguientes centroides a explorar son: " + str(explorable_prototypes[:, 0]))


        for i in range(0, len(explorable_prototypes)):

            centroid = explorable_prototypes[i]
            centroid_layer = current_layer-1
            centroid_inheritage = inheritage + [centroid[1]]
            centroid_id = centroid[0]
            centroid_coords = centroid[2]
            centroid_distance_q = centroid[3]

            explore_centroid(punto_buscado, centroid_layer, centroid_inheritage, centroid_id, coords_puntos_capas, puntos_capas, grupos_capa, n_centroides, metrica, radio, neighbours)


def recursive_approximate_knn_search(n_capas, n_centroides, punto_buscado, vector_original, k_vecinos, metrica,
                           grupos_capa, puntos_capa, labels_capa, dimensiones, radio):
    """
    Performs an approximate k-nearest neighbors (A-KNN) search using a hierarchical tree structure
    and a search radius.

    Parameters:
    n_capas : int
        Number of layers in the hierarchical tree.
    n_centroides : int
        Number of centroids in each group.
    punto_buscado : numpy.ndarray
        The query point for which the nearest neighbors are to be found.
    vector_original : numpy.ndarray
        The original dataset of points.
    k_vecinos : int
        Number of nearest neighbors to find.
    metrica : str
        The distance metric to use for calculating distances.
    grupos_capa : list
        Number of points in each group at each layer.
    puntos_capa : list
        Cluster centroids at each layer.
    labels_capa : list
        Labels assigned to each point at each layer.
    dimensiones : int
        Number of dimensions of the points.
    radio : float
        Search radius for the approximate search.

    Returns:
    list: A list containing three elements:
        - numpy.ndarray: Indices of the k nearest neighbors.
        - numpy.ndarray: Coordinates of the k nearest neighbors.
        - numpy.ndarray: Distances to the k nearest neighbors.
"""

    # Update the metric name for compatibility with scipy
    if metrica == 'manhattan':
        metrica = 'cityblock'  # scipy cdist requires 'cityblock' instead of 'manhattan'

    # Establish the query point
    # print("El punto de query es: ", punto_buscado)
    punto_buscado = np.reshape(punto_buscado, (1, dimensiones))

    # (At the first level, current layer=n_capas-1 and current_group = grupos_capa[n_layer].size[0]-1 = 0)
    inheritage = [0]

    # We take the top-layer prototypes, including its coordinates and distances to the query point
    top_prototypes = range(0, n_centroides)
    coordinates_top_prototypes = np.vstack(puntos_capa[n_capas-1][:])
    distances_top_prototypes = distance.cdist(np.array(punto_buscado), coordinates_top_prototypes, metric=metrica)[0]

    # We would only explore those prototypes which meets the condition / are within a radius (dist<radius)
    explorable_prototypes = np.where(distances_top_prototypes <= radio)[0]

    # We search for every neighbour by exploring each top-layer prototype that meets the radius condition recursively
    neighbours = []
    for prototype_id in explorable_prototypes:
        explore_centroid(punto_buscado, n_capas, inheritage, prototype_id, puntos_capa, labels_capa, grupos_capa, n_centroides, metrica, radio, neighbours)

    # Once the complete index has been explored:

    # If no neighbours have been found:
    if np.array(neighbours).size == 0:

        print("No se han encontrado vecinos para este punto")

        # Pad the array of close points with None objects until it reaches the size of k neighbors
        # To avoid index out of bounds error
        return [np.empty(k_vecinos, dtype=int), np.empty([k_vecinos, vector_original.shape[1]], dtype=float), np.empty(k_vecinos, dtype=float)]


    # If any neighbour have been found:
    else:

        # We obtain its indices, coordinates and distances
        neighbours_ids = np.array(neighbours)
        neighbours_coords = vector_original[neighbours_ids]
        neighbours_dists = distance.cdist(np.array(punto_buscado), neighbours_coords, metric=metrica)[0]

        # And we store them together into a single structure
        neighbours = np.empty((len(neighbours), 3), object)
        for i in range(len(neighbours)):
            neighbours[i][0] = neighbours_ids[i]
            neighbours[i][1] = neighbours_coords[i]
            neighbours[i][2] = neighbours_dists[i]

        neighbours = np.vstack(neighbours)
        #print(f"Se han encontrado {neighbours.shape[0]} vecinos")

        # Sort them according to their distance to the query point
        sorted_neighbours = neighbours[neighbours[:, 2].argsort()]

        # Create the structures to store the data related to the neighbors
        indices_k_vecinos = np.empty(k_vecinos, dtype=int)
        coords_k_vecinos = np.empty([k_vecinos, vector_original.shape[1]], dtype=float)
        dists_k_vecinos = np.empty(k_vecinos, dtype=float)

        # Select the minimum value between k_vecinos and the number of neighbours founded
        minimum = min(k_vecinos, sorted_neighbours.shape[0])

        # Select the k closest points as neighbors
        for i in range(minimum):
            indices_k_vecinos[i] = sorted_neighbours[i][0]
            coords_k_vecinos[i, :] = sorted_neighbours[i][1]
            dists_k_vecinos[i] = sorted_neighbours[i][2]

        # Print them
        print(f"Los vecinos son: {indices_k_vecinos} con distancias {dists_k_vecinos}")

        # And return the results
        return [indices_k_vecinos, coords_k_vecinos, dists_k_vecinos]

