SELECT nom_produit, fournisseur, categorie, unites_stock, quantite
FROM produits
WHERE (fournisseur_pays = 'France' OR categorie IN ('Boissons', 'Desserts'))
AND (quantite LIKE '%boîtes%' OR quantite LIKE '%carton%');
