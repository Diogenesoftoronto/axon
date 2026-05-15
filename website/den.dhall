-- Altum website — static site served via Python http.server on port 8080

let Types = /home/diogenes/Projects/den/dhall/Types.dhall

let Defaults = /home/diogenes/Projects/den/dhall/default.dhall

in  { name = "den-altum"
    , backend = Types.Backend.Railway
    , dockerfile = None Text
    , restartPolicy = Some Types.RestartPolicy.Always
    , healthcheck = None Types.Healthcheck
    , ports = [ { port = 8080, protocol = Some "tcp" } ]
    , volumes = Defaults.defaultVolumes
    , resources = None Types.Resource
    , secrets =
        [ Types.Secret.FromEnv
            { name = "TAILSCALE_AUTHKEY", envVar = "TAILSCALE_AUTHKEY" }
        ]
    , guix = None Types.GuixConfig
    , nix = Some
      { packages =
          [ { name = "python3", version = None Text }
          , { name = "fish",    version = None Text }
          , { name = "git",     version = None Text }
          ]
      , extraConfig = None Text
      }
    , environment =
        [ { mapKey = "DEN_NAME",    mapValue = "den-altum" }
        , { mapKey = "DEN_BACKEND", mapValue = "railway" }
        ]
    , domains = [] : List Text
    }
