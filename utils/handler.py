from django import forms
from optirodig.models import Handler, SchrottWerkstoffe

from optirodig.forms.bootstrap import BootstrapForm


class HandlerModelForm(BootstrapForm, forms.ModelForm):
    """
    Handler Model Form
    """
    class Meta:
        model = Handler
        fields = '__all__'


class UploadWerkstoffeDateiModelForm(BootstrapForm, forms.ModelForm):
    """
    Upload Werkstoffe Datei Model Form
    """
    class Meta:
        model = SchrottWerkstoffe
        fields = '__all__'