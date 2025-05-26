ALTER TABLE ufw22_nodemap ADD COLUMN ip_type TEXT;

UPDATE ufw22_nodemap
SET ip_type = CASE 
  WHEN ip::inet << ANY (ARRAY[
    '10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16',
    '127.0.0.0/8', '169.254.0.0/16',
    '::1/128', 'fc00::/7', 'fe80::/10'
  ]::inet[]) THEN 'Internal'

  WHEN ip::inet << ANY (ARRAY[
    '224.0.0.0/4', 'ff00::/8'
  ]::inet[]) THEN 'Multicast'

  WHEN ip::inet = '255.255.255.255' THEN 'Broadcast'

  ELSE 'External'
END;