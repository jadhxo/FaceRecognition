# FaceRecognition

This project focuses on optimizing the EigenFace algorithm for facial recognition by utilizing parallel computing techniques to enhance performance. The algorithm involves reducing the dimensionality of face data to identify key features, and parallelization helps scale the algorithm for larger datasets and real-time applications, such as security systems. The study explores three parallel computing methods: OpenMP, MPI, and CUDA C. Results show that while OpenMP provides moderate speedup with diminishing returns, MPI offers stable scalability, and CUDA C outperforms both in terms of speedup and efficiency, making it the most effective solution for GPU-based parallel processing.
