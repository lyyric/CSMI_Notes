SELECT
    societe,
    ville,
    pays
FROM
    fournisseurs;


select 1.56, ceil(1.009), floor(1.99),
       round(1.56),round(1.46),
       round(1.563,2),round(1.465,2),
       round(1563,-2),round(1465,-1),
       trunc(1.56),trunc(1.46)
from 
    dual;


SELECT 
    NOM, 
    PRENOM, 
    FLOOR(SALAIRE / 20) AS Salaire_Journalier_Arrondi
FROM 
    EMPLOYES;

SELECT 
    NOM, 
    PRENOM, 
    CEIL(SALAIRE / 20) AS Salaire_Journalier_Arrondi
FROM 
    EMPLOYES;

SELECT 
    NOM_PRODUIT, 
    UNITES_STOCK, 
    PRIX_UNITAIRE, 
    ROUND(UNITES_STOCK * PRIX_UNITAIRE, -2) AS Valeur_Stock_Arrondie
FROM 
    PRODUITS;

SELECT 
    NOM_PRODUIT, 
    UNITES_STOCK, 
    PRIX_UNITAIRE, 
    FLOOR(UNITES_STOCK * PRIX_UNITAIRE / 10) * 10 AS Valeur_Stock_Arrondie
FROM 
    PRODUITS;

SELECT 
    NOM, 
    PRENOM, 
    ROUND((SALAIRE * 12 + COMMISSION), -2) AS Revenu_Annuel_Arrondi
FROM 
    EMPLOYES;


select sysdate, 
       to_char(sysdate,'d dd ddd mm mon month yyyy'),
       to_char(sysdate+0.5,'hh hh24 mi ss sssss'),
       to_date('01/12/2023','mm/dd/yyyy')
from dual;

select sysdate, 
       to_char(sysdate,'d dd ddd mm mon month yyyy'),
       to_char(sysdate+0.5,'hh hh24 mi ss sssss'),
       to_date('01/12/2023','mm/dd/yyyy')
from dual;

select 
    sysdate+4+31, 
    add_months(sysdate+35,4),
    trunc(months_between(sysdate,date_naissance)),
    last_day(sysdate),
    next_day(sysdate,1)
from employes;


SELECT 
    SUM(UNITES_STOCK * PRIX_UNITAIRE) AS Valeur_Stock,
    SUM(UNITES_COMMANDEES * PRIX_UNITAIRE) AS Valeur_Commande
FROM 
    PRODUITS;

select sum(quantite*prix_unitaire) valeur_produits,
       sum((quantite*prix_unitaire)*(1 - remise/100))  ca
from details_commandes;
