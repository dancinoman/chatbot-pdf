
#Translate package to right format for streamlit

with open('requirements.txt', 'r') as fr:
    package_list = fr.readlines()

package_list = [package.replace('==','=') for package in package_list]
print(package_list)

with open('packages.txt', 'w') as fw:
    fw.write(''.join(package_list))
