select pr.nom_produit, fo.societe, cat.NOM_CATEGORIE, pr.unites_stock ,  pr.quantite
from produits pr
join CATEGORIES cat on cat.CODE_CATEGORIE = pr.CODE_CATEGORIE
join FOURNISSEURS fo on fo.NO_FOURNISSEUR = pr.NO_FOURNISSEUR
where (fo.pays = 'France' or 
        cat.NOM_CATEGORIE = 'Boissons' or cat.NOM_CATEGORIE = 'Desserts')
        and (pr.QUANTITE like '%boîtes%' or pr.QUANTITE like '%carton%');




SELECT 
    adresses.pays,
    clients.SOCIETE AS societe_cliente, 
    acheteurs.nom||' '||acheteurs.prenom acheteur, 
    extract(year from commandes.DATE_COMMANDE) annee,
    COUNT(commandes.NO_COMMANDE) AS nombre_commandes
FROM 
    commandes
JOIN 
    acheteurs ON commandes.NO_ACHETEUR = acheteurs.NO_ACHETEUR 
JOIN 
    adresses ON acheteurs.NO_Adresse = adresses.NO_adresse
JOIN 
    clients ON adresses.code_client = clients.CODE_CLIENT 
WHERE 
    extract(year from commandes.DATE_COMMANDE) = 2019
GROUP BY 
    adresses.pays,clients.SOCIETE,acheteurs.nom,acheteurs.prenom,extract(year from commandes.DATE_COMMANDE)
HAVING 
    COUNT(commandes.NO_COMMANDE) > 23;



SELECT 
    adresses.pays,
    clients.SOCIETE AS societe_cliente, 
    employes.nom||' '||employes.prenom vendeur, 
    extract(year from commandes.DATE_COMMANDE) annee,
    COUNT(commandes.NO_COMMANDE) AS nombre_commandes
FROM 
    commandes
JOIN 
    acheteurs ON commandes.NO_ACHETEUR = acheteurs.NO_ACHETEUR 
JOIN 
    adresses ON acheteurs.NO_Adresse = adresses.NO_adresse
JOIN 
    clients ON adresses.code_client = clients.CODE_CLIENT 
JOIN 
    vendeurs ON vendeurs.no_vendeur = commandes.no_vendeur
JOIN 
    employes ON vendeurs.no_vendeur = employes.no_employe    
WHERE 
    extract(year from commandes.DATE_COMMANDE) = 2019
GROUP BY 
    adresses.pays,clients.SOCIETE,employes.nom,employes.prenom,extract(year from commandes.DATE_COMMANDE)
HAVING 
    COUNT(commandes.NO_COMMANDE) > 23;



select vendeurs.pays,
       employes.nom||' '||employes.prenom vendeur,
       sum(port)      
from employes
     join vendeurs on employes.no_employe = vendeurs.no_vendeur
     join commandes USING(no_vendeur)
     join details_commandes USING(no_commande)
where extract(year from  commandes.date_commande)=2019
  and extract(month from  commandes.date_commande)=5
group by vendeurs.pays,employes.nom, employes.prenom     
having sum(port)> 80000;


SELECT em1.nom||' '||em1.prenom as employés_A, em1.FONCTION, em2.nom||' '||em2.prenom as employés_B_gérés_par_A, em3.nom||' '||em3.prenom as employés_C_gérés_par_B
from EMPLOYES em1
left join employes em2 on em1.NO_EMPLOYE = em2.REND_COMPTE
left join employes em3 on em2.NO_EMPLOYE = em3.REND_COMPTE
order by 2, 1, 3, 4;
