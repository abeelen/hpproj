from django.shortcuts import render
from django import forms

import os
from WebServices.settings import BASE_DIR

import cross_match
import cutsky

cross_match_ini=os.path.join(BASE_DIR, "xmatch/data/cross_match_MD.ini")

# Create your views here.

class XmatchForm(forms.Form):

    COORDFRAME_CHOICE = (('galactic', 'Galactic'), ('fk5', 'Equatorial'))
    coordframe = forms.ChoiceField(widget=forms.RadioSelect, choices=COORDFRAME_CHOICE, initial='galactic')
    lon = forms.FloatField(label="lon", initial=6.7, min_value=0, max_value=360)
    lat = forms.FloatField(label="lat", initial=30.45, min_value=-90, max_value=90)
    pixel_number= forms.FloatField(label="pixel number", initial=256, min_value=0)
    pixel_size= forms.FloatField(label="pixel size (arcmin)", initial=1, min_value=0)

    

def index(request):

    if request.method == "POST":
        form = XmatchForm(request.POST)
        if form.is_valid():
            # Do the magic here
            lon = form.cleaned_data.get('lon')
            lat = form.cleaned_data.get('lat')
            coordframe = form.cleaned_data.get('coordframe')
            pixel_number = form.cleaned_data.get('pixel_number')
            pixel_size =form.cleaned_data.get('pixel_size')
            result_xmatch = cross_match.cross_match(cross_match_ini, lonlat=[lon,lat],coordframe=coordframe)
            result_maps = cutsky.cut_sky( lonlat=[lon,lat],patch=[pixel_number,pixel_size],coordframe=coordframe)


            


    else:
        form = XmatchForm()
        result_xmatch = None
        result_maps = None

    return render(
        request,
        'xmatch/index.html',
        {
            'form': form,
            'xmatch': result_xmatch,
            'maps': result_maps
        }
    )
