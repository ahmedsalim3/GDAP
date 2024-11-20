# Frequently Asked Questions (FAQ)

Welcome to the BI ML Disease Prediction app! Below you'll find answers to some common questions to help you navigate and use the application effectively.

### How does the app work?

1.
2.

### What are the different pages in the app?

1.
2.

### How can I use this app?

1.
2.

### What are graph-based embeddings?

Graph-based embeddings represent the structure and relationships within a graph. In this app, these embeddings are generated using algorithms like **Node2Vec**, which captures the patterns in the graph and transforms them into feature vectors. These vectors are then used in machine learning models to predict gene-disease associations.

### How does the app handle negative edges?

Negative edges in the graph are generated using data from the **Protein-Protein Interaction (PPI) database**. Typically, these are created at a **10:1** ratio with positive edges derived from Open Targets. These negative edges are used to represent non-associated gene-disease pairs, providing a balanced dataset for the models may produce better results.

### Can I tune the model’s hyperparameters?

Yes, the app supports **hyperparameter tuning** on the **Embedding/Model Selection** page. You can adjust parameters to optimize the performance of the machine learning models based on the data you're working with.

### What performance metrics are available for model evaluation?

The app provides a variety of metrics to evaluate the performance of your machine learning models, including:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **AUC-ROC**
- **Confusion Matrix**
- **Precision-Recall Curve**

These metrics help assess the model's ability to predict gene-disease associations effectively.

### Can I make predictions for both known and novel gene-disease associations?

Yes, the app allows you to make predictions on both **known** and **novel** gene-disease associations. You can adjust the **prediction threshold** to explore different types of associations, from the most likely to the least likely ones.

### Who developed this app?

This app was developed by the **Stem-Away, July 2024 Batch**. For more details, you can visit the **About Us** page.

### How do I get support if I have more questions?

If you have more questions or need assistance, feel free to reach out via the contact form provided in the app, or consult the **About Us** page for additional contact details.

---

Thank you for using the BI ML Disease Prediction app!
