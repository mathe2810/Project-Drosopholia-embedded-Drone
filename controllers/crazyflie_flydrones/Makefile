# Copyright 1996-2024 Cyberbotics Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Webots Makefile system pour le contrôleur Python crazyflie_flydrones
#
# Vous pouvez ajouter des définitions de variables ici pour personnaliser le build.
# Voir la documentation dans $(WEBOTS_HOME_PATH)/resources/Makefile.include

# Aucun fichier C à compiler pour ce contrôleur Python.
# Le Makefile.include de Webots gère les contrôleurs Python automatiquement.

null :=
space := $(null) $(null)
WEBOTS_HOME_PATH?=$(subst $(space),\ ,$(strip $(subst \,/,$(WEBOTS_HOME))))
include $(WEBOTS_HOME_PATH)/resources/Makefile.include