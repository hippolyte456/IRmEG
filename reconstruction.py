#! /usr/bin/env python


"""
Reconstructeur de sources MEG basé sur MNE-Python.

Ce script permet de reconstruire les sources MEG à partir des données prétraitées
et enregistrées en BIDS. Il réalise les étapes suivantes :
1. Chargement des données evoked (.fif)
2. Calcul de l'espace source
3. Calcul de la solution de champ avancé (forward solution)
4. Application de l'opérateur inverse pour obtenir les sources
5. Sauvegarde des résultats

### Prérequis avant utilisation :
- Avoir installé MNE-Python et mne-bids
- Avoir prétraité les données MEG et généré :
  - Les fichiers evoked (.fif)
  - Les fichiers de covariance du bruit (.fif)
  - Les fichiers de transformation tête-MRI (.fif)
  - Un modèle BEM et un espace source générés avec FreeSurfer
- Avoir un dossier structuré en BIDS avec les sous-dossiers appropriés

### Exemple d'exécution :
```bash
python reconstructer.py sub-01 ses-01
```
"""

import os
import argparse
import pdb

from typing import List

from mne_bids import BIDSPath
import mne
from mne.minimum_norm import (make_inverse_operator, apply_inverse, write_inverse_operator)
from mne import read_cov

from mne.inverse_sparse import mixed_norm 
import numpy as np
import nibabel as nib

 
class reconstructer:
    
    def __init__(self, sub, bids_path, subjects_dir):
        #TODO ALL HARD CODING ...from config file 
        # forward solution
        self.sub = sub
        self.subject = f'sub-{self.sub}'
        self.bids_path = bids_path
        self.subjects_dir = subjects_dir
        self.conductivity = (0.3,)  # for single layer – used in MEG
        # inverse operator
        self.method = "dSPM"
        self.snr = 3.
        self.lambda2 = 1. / self.snr ** 2
    
    #TODO test this extraction
    def _extract_fmri(self, fmri_path: BIDSPath):
        """Extrait une carte de poids fMRI depuis un fichier NIfTI 
        et la projette sur les sources MEG."""
        
        if fmri_path is None:
            raise Exception('fmri_path not set')

        # Charger l'image fMRI au format NIfTI
        img = nib.load(fmri_path.fpath)  
        fmri_data = img.get_fdata()  # Récupérer les valeurs du volume

        # Vérifier si c'est un volume 3D (statique) ou 4D (dynamique)
        if fmri_data.ndim == 4:
            fmri_data = np.mean(fmri_data, axis=-1)  # Moyenne sur le temps

        # Récupérer les sources MEG
        src = mne.read_source_spaces(self.src_fname)

        # Projection fMRI → espace source MEG (utilise un atlas ou un algorithme de projection)
        self.fmri_space = mne.extract_label_time_course(fmri_data, src, mode='auto')

        # Transformer en poids pour `mixed_norm()` (vecteur 1D)
        self.weights = np.ravel(self.fmri_space)  # Assure que ça correspond à `n_sources`

        assert self.weights.shape[0] == src[0]['nuse'], "Mismatch between fMRI and MEG source spaces"
        
        return self.weights
            
    
    def _path_settings(self,evoked_to_process:BIDSPath):
        # INPUTS
        self.evo_fname = evoked_to_process.fpath
        self.trans_fname = evoked_to_process.copy().update(suffix='epo-trans', extension='.fif').fpath #TODO as we need only one transfile for a whole session... BIDS name has to be set accordingly
        self.cov_fname = evoked_to_process.copy().update(suffix='cov', extension='.fif').fpath
         
        # OUTPUTS
        self.src_fname = evoked_to_process.copy().update(suffix=self.event_type + '-' + self.evoked_type + '-source', extension='.fif').fpath  #TODO source should be done one time by session ?!
        self.stc_fname = evoked_to_process.copy().update(suffix=self.event_type + '-' + self.evoked_type + '-estimate', extension=None).fpath 
        
        
    def _compute_source_space(self):
        print("Computing source space...")
        src = mne.setup_source_space(subject=self.subject, spacing='oct6', subjects_dir=self.subjects_dir, add_dist=False) 
        return src

  
    def _compute_forward(self, src):
        print("Computing forward solution...")

        model = mne.make_bem_model(subject=self.subject, ico=4, conductivity=self.conductivity, subjects_dir=self.subjects_dir)
        bem_sol = mne.make_bem_solution(model)
        fwd = mne.make_forward_solution(info=self.info, trans=self.trans_fname, src=src, bem=bem_sol,
                                        meg=True, # include MEG channels
                                        eeg=False, # exclude EEG channels
                                        mindist=5.0, # ignore sources <= 5mm from inner skull
                                        n_jobs=1) # number of jobs to run in parallel
        fwd = mne.convert_forward_solution(fwd, surf_ori=True)
        # Restrict forward solution as necessary for MEG
        fwd = mne.pick_types_forward(fwd, meg=True, eeg=False)
        return fwd
 
 
    def _compute_inverse_operation(self,fwd, noise_cov, use_fmri_prior=False):
        print("Inverse operator...")
        inverse_operator = make_inverse_operator(self.info, fwd, noise_cov, loose=0.2, depth=0.8)
        if not use_fmri_prior:
            stc = apply_inverse(self.evoked, inverse_operator, self.lambda2,
                                method=self.method, pick_ori=None)
        else:
            weights = None  # Remplace par un vecteur numpy de taille (n_sources,)
            # TODO test ...weights = np.ones(n_sources)
            # Utilisation de MxNE avec pondération
            alpha = 50  # Hyperparamètre de régularisation
            stc = mixed_norm(self.evoked, fwd, noise_cov, alpha=alpha, 
                     loose=0.2, depth=0.8, weights=weights)
        return stc    


    def _saving(self, src, stc):
        print("Saving...")
        #TODO overwrite ?
        mne.write_source_spaces(self.src_fname, src) # Sauvegarder le src : OUTPUT 2
        stc.save(self.stc_fname) # Sauvegarder le STC : OUTPUT 3
        #TODO save les configs avec !
  
    
    def reconstruct(self) -> None:
        #TODO enlever les self des lignes suivantes
        if self.evoked_type == 'ERP':
            self.evoked = mne.read_evokeds(self.evo_fname, condition=self.event_type)
        elif self.evoked_type == 'contrast':
            self.evoked = mne.read_evokeds(self.evo_fname, condition=self.event_type)
        else: 
            raise Exception('evoked_type not implemented')
        self.noise_cov = read_cov(self.cov_fname)
        self.info = self.evoked.info

        ###------------- SOURCE SPACE -------------####
        src = self._compute_source_space()
        ###------------- FORWARD SOLUTION -------------###
        fwd = self._compute_forward(src)
        ###------------- INVERSE OPERATOR -------------###
        stc = self._compute_inverse_operation(fwd, self.noise_cov) #TODO here : to fmri file to regularize the reconstruction
        
        ###------------- SAVES MNE OBJECTS -------------###
        self._saving(src,stc)

        print("Done.")
 
 
    def reconstruct_all(self):
        self.evoked_type = 'ERP'
        
        self.event_type = 'cue'
        evoked_path = self.bids_path.copy().update(suffix= self.event_type + '-' + self.evoked_type +'-ave', extension='.fif')
        self._path_settings(evoked_path)
        self.reconstruct()
        
        self.event_type = 'response'
        evoked_path = self.bids_path.copy().update(suffix= self.event_type + '-' + self.evoked_type +'-ave', extension='.fif')
        self._path_settings(evoked_path)
        self.reconstruct()
        
        self.event_type = 'feedback'
        evoked_path = self.bids_path.copy().update(suffix= self.event_type + '-' + self.evoked_type +'-ave', extension='.fif')
        self._path_settings(evoked_path)
        self.reconstruct()
        # TODO generalize to all event and evoked type
        
        
       

def main():
    ####------------- set args and paths & configs ----------------####
    parser = argparse.ArgumentParser()
    parser.add_argument('sub', type=str)
    parser.add_argument('ses', type=str)
    args = parser.parse_args()
    sub = args.sub
    ses = args.ses


    #TODO :from config.yaml !
    BIDS_DIR = '/home/hdreyfus/nasShare/projects/EXPLORE_PLUS_dev/rawdata/derivatives'
    MBP_dir = os.path.join(BIDS_DIR, 'mne_bids_pipeline')
    subjects_dir = os.path.join(BIDS_DIR, 'freesurfer')
    bids_path = BIDSPath(root=MBP_dir, subject=sub, session=ses, datatype='meg', task='EXPLORE', processing='clean', check=False)
    
    print("Starting load data...")
    recon = reconstructer(sub, bids_path, subjects_dir)
    recon.reconstruct_all()
  



if __name__ == '__main__':
    main()
  

