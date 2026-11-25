{{- define "api.name" -}}claim-triage-api{{- end -}}

{{- define "api.fullname" -}}{{ include "api.name" . }}-{{ .Values.track }}{{- end -}}

{{- define "api.selectorLabels" -}}
app.kubernetes.io/name: {{ include "api.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
track: {{ .Values.track }}
{{- end -}}

{{- define "api.labels" -}}
{{ include "api.selectorLabels" . }}
app.kubernetes.io/version: {{ .Values.image.tag | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end -}}
