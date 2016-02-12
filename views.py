from django.shortcuts import render
from django import forms

import cross_match

cross_match_ini="/home/abeelen/Python/WebService/WebServices/xmatch/data/cross_match_MD.ini"

# Create your views here.

class XmatchForm(forms.Form):

    COORDFRAME_CHOICE = (('galactic', 'Galactic'), ('fk5', 'Equatorial'))
    coordframe = forms.ChoiceField(widget=forms.RadioSelect, choices=COORDFRAME_CHOICE, initial='galactic')
    lon = forms.FloatField(label="lon", initial=6.7, min_value=0, max_value=360)
    lat = forms.FloatField(label="lat", initial=30.45, min_value=-90, max_value=90)

    pixel_number= forms.FloatField(label="pixel number", initial=256, min_value=0)
    pixel_size= forms.FloatField(label="pixel size", initial=1, min_value=0)


def index(request):

    if request.method == "POST":
        form = XmatchForm(request.POST)
        if form.is_valid():
            # Do the magic here
            lon = form.cleaned_data.get('lon')
            lat = form.cleaned_data.get('lat')
            coordframe = form.cleaned_data.get('coordframe')
            result_xmatch = cross_match.cross_match(cross_match_ini, lonlat=[lon,lat],coordframe=coordframe)

            print result_xmatch


    else:
        form = XmatchForm()
        result_xmatch = None

    return render(
        request,
        'xmatch/index.html',
        {
            'form': form,
            'result': result_xmatch,
        }
    )
