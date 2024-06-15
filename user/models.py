from django.db import models

class Dimcars(models.Model):
    car_pk = models.AutoField(primary_key=True)
    matricule = models.CharField(unique=True, max_length=255)
    car_owner = models.CharField(max_length=255, blank=True, null=True)
    modele = models.CharField(max_length=255, blank=True, null=True)
    car_type = models.CharField(max_length=45, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'dimcars'


class Dimclients(models.Model):
    client_pk = models.AutoField(primary_key=True)
    code_client = models.CharField(unique=True, max_length=50)
    client_name = models.CharField(max_length=255, blank=True, null=True)
    pick_up_date = models.DateField(db_column='Pick_Up_Date', blank=True, null=True)  # Field name made lowercase.
    pick_up_time = models.TimeField(db_column='Pick_Up_Time', blank=True, null=True)  # Field name made lowercase.
    drop_off_date = models.DateField(db_column='Drop_Off_Date', blank=True, null=True)  # Field name made lowercase.
    drop_off_time = models.TimeField(db_column='Drop_Off_Time', blank=True, null=True)  # Field name made lowercase.
    arrivals_date = models.DateField(db_column='Arrivals_Date', blank=True, null=True)  # Field name made lowercase.
    departure_date = models.DateField(db_column='Departure_Date', blank=True, null=True)  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 'dimclients'


class Dimconcurrents(models.Model):
    concurrent_pk = models.AutoField(primary_key=True)
    concurrent_code = models.CharField(unique=True, max_length=50)
    concurrent_name = models.CharField(max_length=50, blank=True, null=True)
    emplacement = models.CharField(max_length=50, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'dimconcurrents'


class Dimdates(models.Model):
    date_pk = models.AutoField(primary_key=True)
    date = models.DateField(unique=True, blank=True, null=True)
    jour_mois_annee = models.CharField(db_column='Jour_Mois_Annee', max_length=50, blank=True, null=True)  # Field name made lowercase.
    annee = models.IntegerField(db_column='Annee', blank=True, null=True)  # Field name made lowercase.
    id_semestre = models.CharField(max_length=45, blank=True, null=True)
    semestre = models.CharField(max_length=45, blank=True, null=True)
    id_trimestre = models.CharField(max_length=45, blank=True, null=True)
    trimestre = models.CharField(max_length=45, blank=True, null=True)
    id_mois = models.IntegerField(blank=True, null=True)
    mois = models.IntegerField(db_column='Mois', blank=True, null=True)  # Field name made lowercase.
    lib_mois = models.CharField(max_length=45, blank=True, null=True)
    jour = models.IntegerField(blank=True, null=True)
    id_lib_jour = models.IntegerField(blank=True, null=True)
    lib_jour = models.CharField(max_length=45, blank=True, null=True)
    semaine = models.IntegerField(blank=True, null=True)
    jourdeannee = models.IntegerField(db_column='JourDeAnnee', blank=True, null=True)  # Field name made lowercase.
    jour_mois_lettre = models.CharField(db_column='Jour_mois_lettre', max_length=56, blank=True, null=True)  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 'dimdates'


class Dimdestinations(models.Model):
    destination_pk = models.AutoField(primary_key=True)
    destination_code = models.CharField(unique=True, max_length=50)
    destination = models.CharField(max_length=255, blank=True, null=True)
    region_fk = models.ForeignKey('Dimregions', models.DO_NOTHING, db_column='region_fk')

    class Meta:
        managed = False
        db_table = 'dimdestinations'


class Dimdrivers(models.Model):
    driver_pk = models.AutoField(primary_key=True)
    code_driver = models.CharField(unique=True, max_length=255)
    driver_name = models.CharField(max_length=255, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'dimdrivers'


class Dimpartners(models.Model):
    partner_pk = models.AutoField(primary_key=True)
    code_partner = models.CharField(unique=True, max_length=50)
    partner_name = models.CharField(max_length=50, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'dimpartners'


class Dimregions(models.Model):
    region_pk = models.AutoField(primary_key=True)
    codepays = models.CharField(db_column='CodePays', unique=True, max_length=50)  # Field name made lowercase.
    pays = models.CharField(db_column='Pays', max_length=255, blank=True, null=True)  # Field name made lowercase.
    codepostal = models.CharField(db_column='CodePostal', max_length=50, blank=True, null=True)  # Field name made lowercase.
    ville = models.CharField(db_column='Ville', max_length=50, blank=True, null=True)  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 'dimregions'


class Dimrequesttypes(models.Model):
    req_type_pk = models.AutoField(primary_key=True)
    code_req = models.CharField(unique=True, max_length=50)
    req_type = models.CharField(max_length=50, blank=True, null=True)

    class Meta:
        managed = False

class CustomUser(models.Model):
    email = models.EmailField(unique=True)
    first_name = models.CharField(max_length=150)
    last_name = models.CharField(max_length=150)
    password = models.CharField(max_length=150)
    role = models.CharField(max_length=100, null=True, blank=True)  


class Factrequest(models.Model):
    request_pk = models.AutoField(primary_key=True)
    pick_up_date_fk = models.ForeignKey(Dimdates, models.DO_NOTHING, db_column='Pick_Up_Date_fk', to_field='date', blank=True, null=True)  # Field name made lowercase.
    drop_off_date_fk = models.ForeignKey(Dimdates, models.DO_NOTHING, db_column='Drop_Off_Date_fk', to_field='date', related_name='factrequest_drop_off_date_fk_set', blank=True, null=True)  # Field name made lowercase.
    arrivals_date_fk = models.ForeignKey(Dimdates, models.DO_NOTHING, db_column='Arrivals_Date_fk', to_field='date', related_name='factrequest_arrivals_date_fk_set', blank=True, null=True)  # Field name made lowercase.
    departure_date_fk = models.ForeignKey(Dimdates, models.DO_NOTHING, db_column='Departure_Date_fk', to_field='date', related_name='factrequest_departure_date_fk_set', blank=True, null=True)  # Field name made lowercase.
    client_fk = models.ForeignKey(Dimclients, models.DO_NOTHING, db_column='client_fk', blank=True, null=True)
    car_fk = models.ForeignKey(Dimcars, models.DO_NOTHING, db_column='car_fk', blank=True, null=True)
    driver_fk = models.ForeignKey(Dimdrivers, models.DO_NOTHING, db_column='driver_fk', blank=True, null=True)
    destination_fk = models.ForeignKey(Dimdestinations, models.DO_NOTHING, db_column='destination_fk', blank=True, null=True)
    concurrent_fk = models.ForeignKey(Dimconcurrents, models.DO_NOTHING, db_column='concurrent_fk', blank=True, null=True)
    req_fk = models.ForeignKey(Dimrequesttypes, models.DO_NOTHING, db_column='req_fk', blank=True, null=True)
    partner_fk = models.ForeignKey(Dimpartners, models.DO_NOTHING, db_column='partner_fk', blank=True, null=True)
    nbr_extra_hours = models.IntegerField(blank=True, null=True)
    price_charged = models.FloatField(blank=True, null=True)
    price_extra_hours = models.FloatField(blank=True, null=True)
    tva = models.FloatField(blank=True, null=True)
    total_amount = models.FloatField(blank=True, null=True)
    charges = models.FloatField(blank=True, null=True)
    profit = models.FloatField(blank=True, null=True)
    nbr_cars = models.IntegerField(blank=True, null=True)
    comission = models.FloatField(blank=True, null=True)
    comission_ammount = models.FloatField(blank=True, null=True)
    partner_cost = models.FloatField(blank=True, null=True)
    consommation_price = models.FloatField(blank=True, null=True)
    car_price = models.FloatField(blank=True, null=True)
    consommation_moy_carburant_l_par_100km = models.FloatField(db_column='Consommation_moy_carburant_L_par_100km', blank=True, null=True)  # Field name made lowercase.
    emission_co2 = models.FloatField(db_column='Emission_CO2', blank=True, null=True)  # Field name made lowercase.
    available_days = models.IntegerField(blank=True, null=True)
    available_hours = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'factrequest'

def __str__(self):
    return self.name
