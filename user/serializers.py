from rest_framework import serializers
from .models import CustomUser
from .models import Dimcars
from .models import Dimclients
from .models import Dimdates
from .models import Dimdrivers
from .models import Dimpartners
from .models import Dimdestinations
from .models import Dimdates
from .models import Dimregions
from .models import Dimrequesttypes
from .models import Factrequest
class CustomerUserSerializer(serializers.ModelSerializer):
    class Meta:
        model = CustomUser
        fields = '__all__'

class DimcarsSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dimcars
        fields = '__all__'

class DimclientsSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dimclients
        fields = '__all__'

class DimdatesSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dimdates
        fields = '__all__'

class DimdriversSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dimdrivers
        fields = '__all__'

class DimpartnersSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dimpartners
        fields = '__all__'

class DimdestinationsSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dimdestinations
        fields = '__all__'

class DimregionsSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dimregions
        fields = '__all__'

class DimrequesttypesSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dimrequesttypes
        fields = '__all__'

class FactrequestSerializer(serializers.ModelSerializer):
    class Meta:
        model = Factrequest
        fields = '__all__'


