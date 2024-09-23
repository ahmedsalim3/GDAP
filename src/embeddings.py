# inspired from:
# Node2Vec: https://github.com/VHRanger/nodevectors/blob/master/nodevectors/node2vec.py
# ProNE: https://github.com/VHRanger/nodevectors/blob/master/nodevectors/prone.py
# GGVec: https://github.com/VHRanger/nodevectors/blob/master/nodevectors/ggvec.py

import numba
import numpy as np
import pandas as pd
import time
import warnings
import csrgraph as cg
from nodevectors.embedders import BaseNodeEmbedder
import scipy
from scipy import sparse, linalg
from sklearn import preprocessing
from sklearn.utils.extmath import randomized_svd

# Gensim triggers automatic useless warnings for windows users...
warnings.simplefilter("ignore", category=UserWarning)
import gensim
warnings.simplefilter("default", category=UserWarning)


class Node2Vec(BaseNodeEmbedder):
    def __init__(
        self, 
        n_components=32,
        walklen=30, 
        epochs=20,
        return_weight=1.,
        neighbor_weight=1.,
        threads=0, 
        keep_walks=False,
        verbose=True,
        w2vparams={"window":10, "negative":5, "epochs":10,
                   "batch_words":128}):
        """
        Parameters
        ----------
        walklen : int
            length of the random walks
        epochs : int
            number of times to start a walk from each nodes
        threads : int
            number of threads to use. 0 is full use
        n_components : int
            number of resulting dimensions for the embedding
            This should be set here rather than in the w2vparams arguments
        return_weight : float in (0, inf]
            Weight on the probability of returning to node coming from
            Having this higher tends the walks to be 
            more like a Breadth-First Search.
            Having this very high  (> 2) makes search very local.
            Equal to the inverse of p in the Node2Vec paper.
        neighbor_weight : float in (0, inf]
            Weight on the probability of visitng a neighbor node
            to the one we're coming from in the random walk
            Having this higher tends the walks to be 
            more like a Depth-First Search.
            Having this very high makes search more outward.
            Having this very low makes search very local.
            Equal to the inverse of q in the Node2Vec paper.
        keep_walks : bool
            Whether to save the random walks in the model object after training
        w2vparams : dict
            dictionary of parameters to pass to gensim's word2vec
            Don't set the embedding dimensions through arguments here.
            
            window : int, optional
            Maximum distance between the current and predicted word within a sentence.
            
            negative : int, optional
            If > 0, negative sampling will be used, the int for negative specifies how many "noise words"
            should be drawn (usually between 5-20).
            If 0, negative sampling will not be used.
            
            epochs : int, optional
            Number of iterations (epochs) over the corpus. (Formerly: `iter`)
            
            batch_words : int, optional
            Target size (in words) for batches of examples passed to worker threads (and
            thus cython routines).(Larger batches will be passed if individual
            texts are longer than 10000 words, but the standard cython code truncates to that maximum.)
        """
        if type(threads) is not int:
            raise ValueError("Threads argument must be an int!")
        if walklen < 1 or epochs < 1:
            raise ValueError("Walklen and epochs arguments must be > 1")
        self.n_components = n_components
        self.walklen = walklen
        self.epochs = epochs
        self.keep_walks = keep_walks
        if 'size' in w2vparams.keys():
            raise AttributeError("Embedding dimensions should not be set "
                + "through w2v parameters, but through n_components")
        self.w2vparams = w2vparams
        self.return_weight = return_weight
        self.neighbor_weight = neighbor_weight
        if threads == 0:
            threads = numba.config.NUMBA_DEFAULT_NUM_THREADS
        self.threads = threads
        w2vparams['workers'] = threads
        self.verbose = verbose

    def fit(self, G):
        """
        NOTE: Currently only support str or int as node name for graph
        Parameters
        ----------
        G : graph data
            Graph to embed
            Can be any graph type that's supported by csrgraph library
            (NetworkX, numpy 2d array, scipy CSR matrix, CSR matrix components)
        """
        if not isinstance(G, cg.csrgraph):
            G = cg.csrgraph(G, threads=self.threads)
        if G.threads != self.threads:
            G.set_threads(self.threads)
        # Because networkx graphs are actually iterables of their nodes
        #   we do list(G) to avoid networkx 1.X vs 2.X errors
        node_names = G.names
        if type(node_names[0]) not in [int, str, np.int32, np.uint32, 
                                       np.int64, np.uint64]:
            raise ValueError("Graph node names must be int or str!")
        # Adjacency matrix
        walks_t = time.time()
        if self.verbose:
            print("Making walks...", end=" ")
        self.walks = G.random_walks(walklen=self.walklen, 
                                    epochs=self.epochs,
                                    return_weight=self.return_weight,
                                    neighbor_weight=self.neighbor_weight)
        if self.verbose:
            print(f"Done, T={time.time() - walks_t:.2f}")
            print("Mapping Walk Names...", end=" ")
        map_t = time.time()
        self.walks = pd.DataFrame(self.walks)
        # Map nodeId -> node name
        node_dict = dict(zip(np.arange(len(node_names)), node_names))
        for col in self.walks.columns:
            self.walks[col] = self.walks[col].map(node_dict).astype(str)
        # Somehow gensim only trains on this list iterator
        # it silently mistrains on array input
        self.walks = [list(x) for x in self.walks.itertuples(False, None)]
        if self.verbose:
            print(f"Done, T={time.time() - map_t:.2f}")
            print("Training W2V...", end=" ")
        w2v_t = time.time()
        # Train gensim word2vec model on random walks
        self.model = gensim.models.Word2Vec(
            sentences=self.walks,
            vector_size=self.n_components,
            **self.w2vparams)
        if not self.keep_walks:
            del self.walks
        if self.verbose:
            print(f"Done, T={time.time() - w2v_t:.2f}")

    def fit_transform(self, G):
        """
        NOTE: Currently only support str or int as node name for graph
        Parameters
        ----------
        G : graph data
            Graph to embed
            Can be any graph type that's supported by csrgraph library
            (NetworkX, numpy 2d array, scipy CSR matrix, CSR matrix components)
        """
        if not isinstance(G, cg.csrgraph):
            G = cg.csrgraph(G, threads=self.threads)
        self.fit(G)
        node_names = G.names
        w = np.array([self.predict(name) for name in node_names])
        return w
    
    def predict(self, node_name):
        """
        Return vector associated with node
        node_name : str or int
            either the node ID or node name depending on graph format
        """
        # current hack to work around word2vec problem
        # ints need to be str -_-
        if type(node_name) is not str:
            node_name = str(node_name)
        return self.model.wv.__getitem__(node_name)

    def save_vectors(self, out_file):
        """
        Save as embeddings in gensim.models.KeyedVectors format
        """
        self.model.wv.save_word2vec_format(out_file)

    def load_vectors(self, out_file):
        """
        Load embeddings from gensim.models.KeyedVectors format
        """
        self.model = gensim.wv.load_word2vec_format(out_file)
        

class ProNE(BaseNodeEmbedder):
    def __init__(self, n_components=32, step=10, mu=0.2, theta=0.5, 
                exponent=0.75, verbose=True):
        """
        Fast first order, global method.

        Embeds by doing spectral propagation over an initial SVD embedding.
        This can be seen as augmented spectral propagation.

        Parameters :
        --------------
        step : int >= 1
            Step of recursion in post processing step.
            More means a more refined embedding.
            Generally 5-10 is enough
        mu : float
            Damping factor on optimization post-processing
            You rarely have to change it
        theta : float
            Bessel function parameter in Chebyshev polynomial approximation
            You rarely have to change it
        exponent : float in [0, 1]
            Exponent on negative sampling
            You rarely have to change it
        References:
        --------------
        Reference impl: https://github.com/THUDM/ProNE
        Reference Paper: https://www.ijcai.org/Proceedings/2019/0594.pdf
        """
        self.n_components = n_components
        self.step = step
        self.mu = mu
        self.theta = theta
        self.exponent = exponent
        self.verbose = verbose

    
    def fit_transform(self, graph):
        """
        NOTE: Currently only support str or int as node name for graph
        Parameters
        ----------
        nxGraph : graph data
            Graph to embed
            Can be any graph type that's supported by CSRGraph library
            (NetworkX, numpy 2d array, scipy CSR matrix, CSR matrix components)
        """
        G = cg.csrgraph(graph)
        features_matrix = self.pre_factorization(G.mat,
                                                 self.n_components,
                                                 self.exponent)
        vectors = ProNE.chebyshev_gaussian(
            G.mat, features_matrix, self.n_components,
            step=self.step, mu=self.mu, theta=self.theta)
        self.model = dict(zip(G.nodes(), vectors))
        return vectors

    
    def fit(self, graph):
        """
        NOTE: Currently only support str or int as node name for graph
        Parameters
        ----------
        nxGraph : graph data
            Graph to embed
            Can be any graph type that's supported by CSRGraph library
            (NetworkX, numpy 2d array, scipy CSR matrix, CSR matrix components)
        """
        G = cg.csrgraph(graph)
        features_matrix = self.pre_factorization(G.mat,
                                                 self.n_components, 
                                                 self.exponent)
        vectors = ProNE.chebyshev_gaussian(
            G.mat, features_matrix, self.n_components,
            step=self.step, mu=self.mu, theta=self.theta)
        self.model = dict(zip(G.nodes(), vectors))

    @staticmethod
    def tsvd_rand(matrix, n_components):
        """
        Sparse randomized tSVD for fast embedding
        """
        l = matrix.shape[0]
        # Is this csc conversion necessary?
        smat = sparse.csc_matrix(matrix)
        U, Sigma, VT = randomized_svd(smat, 
            n_components=n_components, 
            n_iter=5, random_state=None)
        U = U * np.sqrt(Sigma)
        U = preprocessing.normalize(U, "l2")
        return U

    @staticmethod
    def pre_factorization(G, n_components, exponent):
        """
        Network Embedding as Sparse Matrix Factorization
        """
        C1 = preprocessing.normalize(G, "l1")
        # Prepare negative samples
        neg = np.array(C1.sum(axis=0))[0] ** exponent
        neg = neg / neg.sum()
        neg = sparse.diags(neg, format="csr")
        neg = G.dot(neg)
        # Set negative elements to 1 -> 0 when log
        C1.data[C1.data <= 0] = 1
        neg.data[neg.data <= 0] = 1
        C1.data = np.log(C1.data)
        neg.data = np.log(neg.data)
        C1 -= neg
        features_matrix = ProNE.tsvd_rand(C1, n_components=n_components)
        return features_matrix

    @staticmethod
    def svd_dense(matrix, dimension):
        """
        dense embedding via linalg SVD
        """
        U, s, Vh = linalg.svd(matrix, full_matrices=False, 
                              check_finite=False, 
                              overwrite_a=True)
        U = np.array(U)
        U = U[:, :dimension]
        s = s[:dimension]
        s = np.sqrt(s)
        U = U * s
        U = preprocessing.normalize(U, "l2")
        return U

    @staticmethod
    def chebyshev_gaussian(G, a, n_components=32, step=10, 
                           mu=0.5, theta=0.5):
        """
        NE Enhancement via Spectral Propagation

        G : Graph (csr graph matrix)
        a : features matrix from tSVD
        mu : damping factor
        theta : bessel function parameter
        """
        nnodes = G.shape[0]
        if step == 1:
            return a
        A = sparse.eye(nnodes) + G
        DA = preprocessing.normalize(A, norm='l1')
        # L is graph laplacian
        L = sparse.eye(nnodes) - DA
        M = L - mu * sparse.eye(nnodes)
        Lx0 = a
        Lx1 = M.dot(a)
        Lx1 = 0.5 * M.dot(Lx1) - a
        conv = scipy.special.iv(0, theta) * Lx0
        conv -= 2 * scipy.special.iv(1, theta) * Lx1
        # Use Bessel function to get Chebyshev polynomials
        for i in range(2, step):
            Lx2 = M.dot(Lx1)
            Lx2 = (M.dot(Lx2) - 2 * Lx1) - Lx0
            if i % 2 == 0:
                conv += 2 * scipy.special.iv(i, theta) * Lx2
            else:
                conv -= 2 * scipy.special.iv(i, theta) * Lx2
            Lx0 = Lx1
            Lx1 = Lx2
            del Lx2
        mm = A.dot(a - conv)
        emb = ProNE.svd_dense(mm, n_components)
        return emb
    

class GGVec(BaseNodeEmbedder):
    def __init__(self, 
        n_components=32,
        order=1,
        learning_rate=0.1, max_loss=10.,
        tol="auto", tol_samples=30,
        exponent=0.33,
        threads=0,
        negative_ratio=0.15,
        max_epoch=350, 
        verbose=False):
        """
        GGVec: Fast global first (and higher) order local embeddings.

        This algorithm directly minimizes related nodes' distances.
        It uses a relaxation pass (negative sample) + contraction pass (loss minimization)
        To find stable embeddings based on the minimal dot product of edge weights.

        Parameters:
        -------------
        n_components (int): 
            Number of individual embedding dimensions.
        order : int >= 1
            Meta-level of the embeddings. Improves link prediction performance.
            Setting this higher than 1 ~quadratically slows down algorithm
                Order = 1 directly optimizes the graph.
                Order = 2 optimizes graph plus neighbours of neighbours
                Order = 3 optimizes up to 3rd order edges
                (and so on)
            Higher order edges are automatically weighed using GraRep-style graph formation
            Eg. the higher-order graph is from stable high-order random walk distribution.
        negative_ratio : float in [0, 1]
            Negative sampling ratio.
            Setting this higher will do more negative sampling.
            This is slower, but can lead to higher quality embeddings.
        exponent : float
            Weighing exponent in loss function. 
            Having this lower reduces effect of large edge weights.
        tol : float in [0, 1] or "auto"
            Optimization early stopping criterion.
            Stops average loss < tol for tol_samples epochs.
            "auto" sets tol as a function of learning_rate
        tol_samples : int
            Optimization early stopping criterion.
            This is the number of epochs to sample for loss stability.
            Once loss is stable over this number of epochs we stop early.
        negative_decay : float in [0, 1]
            Decay on negative ratio.
            If >0 then negative ratio will decay by (1-negative_decay) ** epoch
            You should usually leave this to 0.
        max_epoch : int
            Stopping criterion.
        max_count : int
            Ceiling value on edge weights for numerical stability
        learning_rate : float in [0, 1]
            Optimization learning rate.
        max_loss : float
            Loss value ceiling for numerical stability.
        """
        self.n_components = n_components
        self.tol = tol
        self.order=order
        self.max_epoch = max_epoch
        self.learning_rate = learning_rate
        self.exponent = exponent
        self.max_loss = max_loss
        self.tol_samples = tol_samples
        self.threads = threads
        self.negative_ratio = negative_ratio
        self.verbose = verbose

    def fit(self, graph):
        """
        NOTE: Currently only support str or int as node name for graph
        Parameters
        ----------
        nxGraph : graph data
            Graph to embed
            Can be any graph type that's supported by CSRGraph library
            (NetworkX, numpy 2d array, scipy CSR matrix, CSR matrix components)
        """
        G = cg.csrgraph(graph, threads=self.threads)
        vectors = G.ggvec(
            n_components=self.n_components, 
            order=self.order,
            exponent=self.exponent,
            tol=self.tol, max_epoch=self.max_epoch,
            learning_rate=self.learning_rate, 
            tol_samples=self.tol_samples,
            max_loss=self.max_loss,
            negative_ratio=self.negative_ratio,
            verbose=self.verbose)
        self.model = dict(zip(G.nodes(), vectors))

    def fit_transform(self, graph):
        """
        NOTE: Currently only support str or int as node name for graph
        Parameters
        ----------
        nxGraph : graph data
            Graph to embed
            Can be any graph type that's supported by CSRGraph library
            (NetworkX, numpy 2d array, scipy CSR matrix, CSR matrix components)
        """
        G = cg.csrgraph(graph, threads=self.threads)
        vectors = G.ggvec(
            n_components=self.n_components, 
            order=self.order,
            exponent=self.exponent,
            tol=self.tol, max_epoch=self.max_epoch,
            learning_rate=self.learning_rate, 
            tol_samples=self.tol_samples,
            max_loss=self.max_loss,
            negative_ratio=self.negative_ratio,
            verbose=self.verbose)
        self.model = dict(zip(G.nodes(), vectors))
        return vectors


def simple_node_embedding(G, dim=64):
    print("Generating simple node embeddings...")
    embeddings = {}
    for node in G.nodes():
        # Use node degree as a feature
        degree = G.degree(node)
        # Use average neighbor degree as another feature
        neighbor_degrees = [G.degree(n) for n in G.neighbors(node)]
        avg_neighbor_degree = np.mean(neighbor_degrees) if neighbor_degrees else 0
        # Create a simple embedding vector
        embedding = np.zeros(dim)
        embedding[0] = degree
        embedding[1] = avg_neighbor_degree
        # Fill the rest with random values (you could add more graph-based features here)
        embedding[2:] = np.random.randn(dim-2)
        embeddings[node] = embedding / np.linalg.norm(embedding)  # Normalize
    
    return embeddings
